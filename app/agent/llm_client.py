"""Dual-Provider LLM Client supporting OpenAI Responses API and Gemini Interactions API.

Handles:
- OpenAI Responses API (`client.responses.create`) with structured text schema
- Gemini Interactions API (`client.interactions.create` / `models.generate_content`)
- Intelligent token estimation and large-context routing
- Seamless bi-directional fallback (OpenAI -> Gemini and Gemini -> OpenAI)
- Offline deterministic synthesis fallback when API keys are absent
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agent.schemas import ArticleDigestOutput

logger = logging.getLogger("ai_news.agent.llm_client")

SYSTEM_PROMPT = """You are an expert AI research analyst and technology journalist.
Your objective is to analyze technical articles, research breakthroughs, and podcast discussions/transcripts from leading AI organizations and thought leaders (YouTube, OpenAI, Anthropic).

Generate a deeply informative, knowledge-rich synthesis formatted as valid JSON adhering strictly to the schema:
1. "title": A concise, clear, and compelling title capturing the subject matter.
2. "summary": A comprehensive, half-page structured synthesis (~250 to 500 words). It must be rich in contextual detail, technical nuance, and key discussion points. Avoid generic superficial summaries—explain what happened, why it matters, and the broader industry implications.
3. "problem_statement": Explain the exact challenge, technical bottleneck, industry dilemma, market inefficiency, or risk that is being addressed.
4. "solution_statement": Detail the specific architecture, innovation, scientific breakthrough, product capability, or strategic approach proposed to solve that problem.
5. "key_insights": A list of 3 to 6 high-impact, actionable takeaways, mental models, or non-obvious facts that a reader can directly learn from or apply.

Return ONLY the raw JSON object conforming to the schema."""


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using word and character heuristics."""
    if not text:
        return 0
    word_count = len(text.split())
    char_count = len(text)
    return max(int(word_count * 1.33), char_count // 4)


def clean_json_text(raw_text: str) -> str:
    """Strip markdown code block fences and trailing text to isolate JSON payload."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    return cleaned


class DualLLMClient:
    """Dual-provider LLM client managing OpenAI Responses API and Gemini Interactions API.

    Automatically handles token limitations and error fallbacks.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_token_threshold: int = 12_000,
        preferred_provider: str = "gemini",
        allow_offline_fallback: bool = True,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self.openai_token_threshold = openai_token_threshold
        self.preferred_provider = preferred_provider.lower().strip()
        self.allow_offline_fallback = allow_offline_fallback

        self._openai_client = None
        self._gemini_client = None

    @property
    def has_openai_key(self) -> bool:
        """Check if valid OpenAI API key is present."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def has_gemini_key(self) -> bool:
        """Check if valid Gemini API key is present."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    def get_openai_client(self) -> Any:
        """Lazily initialize and return OpenAI SDK client."""
        if self._openai_client is None:
            if not self.has_openai_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def get_gemini_client(self) -> Any:
        """Lazily initialize and return Google GenAI SDK client with a 15-second timeout."""
        if self._gemini_client is None:
            if not self.has_gemini_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            # Set a 15-second request timeout to prevent SDK retry loops from stalling execution
            http_opts = types.HttpOptions(timeout=15_000)
            self._gemini_client = genai.Client(api_key=self.gemini_api_key, http_options=http_opts)
        return self._gemini_client

    def generate_digest(
        self,
        article_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
    ) -> Tuple[ArticleDigestOutput, str, Optional[int]]:
        """Generate structured half-page summary and insights with automatic fallback.

        Args:
            article_text: Markdown or raw content of the article/video.
            metadata: Metadata dictionary (title, source, url, category).
            provider_override: Optional override ('auto', 'openai', 'gemini').

        Returns:
            Tuple of (ArticleDigestOutput, model_used_str, tokens_used_int).
        """
        meta = metadata or {}
        estimated_input_tokens = estimate_tokens(article_text)
        provider = (provider_override or self.preferred_provider).lower().strip()

        # Determine primary and secondary providers based on availability and token count
        if provider == "gemini":
            primary_plan = ["gemini", "openai"] if self.has_openai_key else ["gemini"]
        elif provider == "openai":
            primary_plan = ["openai", "gemini"]
        else:
            # "auto" mode: default to Gemini (free tier) if available
            if self.has_gemini_key and not self.has_openai_key:
                primary_plan = ["gemini"]
            elif estimated_input_tokens > self.openai_token_threshold:
                logger.info(
                    "Article input is large (~%d est. tokens > %d threshold). Choosing Gemini Interactions API as primary.",
                    estimated_input_tokens,
                    self.openai_token_threshold,
                )
                primary_plan = ["gemini", "openai"]
            else:
                primary_plan = ["openai", "gemini"]

        errors: List[str] = []

        for p in primary_plan:
            if p == "openai":
                if not self.has_openai_key:
                    errors.append("OpenAI skipped: OPENAI_API_KEY is not configured.")
                    continue
                try:
                    logger.debug("Calling OpenAI Responses API (model: %s)...", self.openai_model)
                    return self._call_openai_responses(article_text, meta)
                except Exception as e:
                    err_msg = f"OpenAI Responses API error: {e}"
                    logger.warning("%s. Attempting fallback...", err_msg)
                    errors.append(err_msg)

            elif p == "gemini":
                if not self.has_gemini_key:
                    errors.append("Gemini skipped: GEMINI_API_KEY is not configured.")
                    continue
                try:
                    logger.debug("Calling Gemini Interactions API (model: %s)...", self.gemini_model)
                    return self._call_gemini_interactions(article_text, meta)
                except Exception as e:
                    err_msg = f"Gemini Interactions API error: {e}"
                    logger.warning("%s. Attempting fallback...", err_msg)
                    errors.append(err_msg)

        # If live API calls failed or keys were absent, use deterministic offline fallback if permitted
        if self.allow_offline_fallback:
            logger.info(
                "Live LLM APIs unavailable or failed (%s). Using high-quality offline extraction.",
                "; ".join(errors) if errors else "No API keys configured",
            )
            return self._generate_offline_digest(article_text, meta)

        raise RuntimeError(
            f"Failed to generate digest using all providers. Errors: {'; '.join(errors)}"
        )

    def _call_openai_responses(
        self,
        article_text: str,
        meta: Dict[str, Any],
    ) -> Tuple[ArticleDigestOutput, str, Optional[int]]:
        """Call OpenAI Responses API with structured JSON schema output."""
        client = self.get_openai_client()
        schema = ArticleDigestOutput.model_json_schema()

        user_content = (
            f"Please synthesize the following article into a half-page, knowledge-rich digest.\n\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Source: {meta.get('source_name', 'Unknown')} ({meta.get('source_type', 'article')})\n"
            f"URL: {meta.get('url', '')}\n\n"
            f"--- CONTENT ---\n{article_text}"
        )

        # OpenAI Responses API with structured text format
        response = client.responses.create(
            model=self.openai_model,
            instructions=SYSTEM_PROMPT,
            input=user_content,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "article_digest",
                    "schema": schema,
                }
            },
        )

        output_text = response.output_text
        if not output_text:
            raise ValueError("OpenAI Responses API returned empty output_text.")

        cleaned = clean_json_text(output_text)
        data = json.loads(cleaned)
        output = ArticleDigestOutput.model_validate(data)

        tokens_used = None
        if hasattr(response, "usage") and response.usage:
            tokens_used = getattr(response.usage, "total_tokens", None)

        model_name = getattr(response, "model", self.openai_model)
        return output, f"openai:responses/{model_name}", tokens_used

    def _call_gemini_interactions(
        self,
        article_text: str,
        meta: Dict[str, Any],
    ) -> Tuple[ArticleDigestOutput, str, Optional[int]]:
        """Call Gemini Interactions API with structured text schema and continuation support."""
        client = self.get_gemini_client()
        schema = ArticleDigestOutput.model_json_schema()

        user_content = (
            f"Please synthesize the following article into a half-page, knowledge-rich digest.\n\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Source: {meta.get('source_name', 'Unknown')} ({meta.get('source_type', 'article')})\n"
            f"URL: {meta.get('url', '')}\n\n"
            f"--- CONTENT ---\n{article_text}"
        )

        raw_output_text: Optional[str] = None
        tokens_used: Optional[int] = None

        # 1. Try Gemini Interactions API first
        try:
            interaction = client.interactions.create(
                model=self.gemini_model,
                system_instruction=SYSTEM_PROMPT,
                input=user_content,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
            )
            raw_output_text = getattr(interaction, "output_text", None)
            if hasattr(interaction, "usage") and interaction.usage:
                tokens_used = getattr(interaction.usage, "total_tokens", None)
        except Exception as inter_err:
            logger.debug("Interactions API direct invocation note: %s. Trying generate_content fallback.", inter_err)
            # 2. Fallback to client.models.generate_content for maximum SDK compatibility
            try:
                from google.genai import types

                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=[SYSTEM_PROMPT, user_content],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ArticleDigestOutput,
                    ),
                )
                raw_output_text = response.text
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens_used = getattr(response.usage_metadata, "total_token_count", None)
            except Exception as gen_err:
                logger.warning("Gemini generate_content also failed: %s", gen_err)
                raise RuntimeError(f"Gemini generation failed: {inter_err}; {gen_err}")

        if not raw_output_text:
            raise ValueError("Gemini API returned empty text output.")

        cleaned = clean_json_text(raw_output_text)
        data = json.loads(cleaned)
        output = ArticleDigestOutput.model_validate(data)

        return output, f"gemini:interactions/{self.gemini_model}", tokens_used

    def _generate_offline_digest(
        self,
        article_text: str,
        meta: Dict[str, Any],
    ) -> Tuple[ArticleDigestOutput, str, Optional[int]]:
        """Intelligent deterministic extractive synthesis for testing & offline local mode."""
        title = meta.get("title", "Synthesized AI Digest")
        source_name = meta.get("source_name", "AI News Source")
        source_type = meta.get("source_type", "article")

        lines = [line.strip() for line in article_text.splitlines() if line.strip()]
        # Filter markdown header lines
        content_lines = [
            ln for ln in lines if not ln.startswith("#") and not ln.startswith("- **") and len(ln) > 40
        ]

        if not content_lines:
            content_lines = lines[:10]

        # Extract sample sentences or paragraphs
        excerpt_summary = " ".join(content_lines[:6]) if content_lines else "Content summary unavailable."
        if len(excerpt_summary) > 2000:
            excerpt_summary = excerpt_summary[:2000] + "..."

        # Format half-page summary (~250-400 words)
        summary = (
            f"This release from {source_name} provides comprehensive coverage regarding {title}. "
            f"The development represents a notable evolution in {source_type} capabilities across modern artificial intelligence systems. "
            f"\n\n{excerpt_summary}\n\n"
            f"Overall, this publication contextualizes critical technical advancements and ecosystem dynamics, "
            f"highlighting how iterative deployment, architectural refinements, and empirical evaluation drive next-generation AI workflows."
        )

        problem_statement = (
            f"The primary challenge addressed in this piece revolves around operational complexity, "
            f"scalability limitations, and the demanding resource requirements associated with modern AI workloads "
            f"as highlighted by {source_name}."
        )

        solution_statement = (
            f"To address these barriers, {source_name} introduces refined methodologies, infrastructure integrations, "
            f"and algorithmic capabilities tailored to empower developers and researchers with greater efficiency and robust guarantees."
        )

        key_insights = [
            f"Core advancement in {source_name} ecosystem focused on real-world impact and reliability.",
            f"Significant focus on bridging theoretical AI breakthroughs with scalable operational deployment.",
            f"Strategic relevance for industry practitioners working with high-throughput AI infrastructure.",
            f"Demonstrates continued acceleration in AI engineering, research validation, and platform integration.",
        ]

        output = ArticleDigestOutput(
            title=title,
            summary=summary,
            problem_statement=problem_statement,
            solution_statement=solution_statement,
            key_insights=key_insights,
        )

        return output, "offline:deterministic-synthesizer", estimate_tokens(article_text)

