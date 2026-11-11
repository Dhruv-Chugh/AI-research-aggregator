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

DEFAULT_GEMINI_FALLBACK_MODELS: List[str] = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
]


def _is_rate_limit_error(err: Exception) -> bool:
    """Return True for quota/rate-limit (429) errors that should fail fast."""
    upper = str(err).upper()
    return any(tok in upper for tok in ("429", "RESOURCE_EXHAUSTED", "TOO MANY REQUESTS"))


def _is_transient_gemini_error(err: Exception) -> bool:
    """Return True for transient overload/timeout/not-found errors (503, 504, deadline exceeded)."""
    upper = str(err).upper()
    return any(
        tok in upper
        for tok in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "TIMED OUT", "TIMEOUT", "NOT_FOUND", "404")
    )


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
        gemini_fallback_models: Optional[List[str]] = None,
        openai_token_threshold: int = 12_000,
        preferred_provider: str = "gemini",
        allow_offline_fallback: bool = True,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        if gemini_fallback_models is not None:
            self.gemini_fallback_models = [m.strip() for m in gemini_fallback_models if m.strip()]
        else:
            env_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
            self.gemini_fallback_models = (
                [m.strip() for m in env_fallbacks.split(",") if m.strip()]
                if env_fallbacks.strip()
                else list(DEFAULT_GEMINI_FALLBACK_MODELS)
            )
        self.last_gemini_model_used: Optional[str] = None
        self._gemini_timeout_ms: int = 0

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

    def get_gemini_client(self, timeout_ms: int = 30_000) -> Any:
        """Lazily initialize and return Google GenAI SDK client with fast-fail retries."""
        if self._gemini_client is None or timeout_ms > self._gemini_timeout_ms:
            if not self.has_gemini_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            # Single attempt to prevent SDK 429 exponential backoff hangs; configurable timeout
            http_opts = types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._gemini_client = genai.Client(api_key=self.gemini_api_key, http_options=http_opts)
            self._gemini_timeout_ms = timeout_ms
            # Interactions sub-client uses Speakeasy RetryConfig instead of http_options.retry_options
            if hasattr(self._gemini_client, "interactions") and hasattr(
                self._gemini_client.interactions, "sdk_configuration"
            ):
                self._gemini_client.interactions.sdk_configuration.retry_config = None
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

    def _invoke_gemini_model(
        self,
        client: Any,
        model_name: str,
        system_instruction: str,
        user_content: str,
        schema: Dict[str, Any],
        schema_model: Any,
        timeout_ms: int,
    ) -> Tuple[str, Optional[int]]:
        """Invoke Interactions API then generate_content fallback for a single Gemini model."""
        # 1. Try Gemini Interactions API first
        try:
            interaction = client.interactions.create(
                model=model_name,
                system_instruction=system_instruction,
                input=user_content,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
            )
            raw_text = getattr(interaction, "output_text", None)
            tokens_used = (
                getattr(interaction.usage, "total_tokens", None)
                if hasattr(interaction, "usage") and interaction.usage
                else None
            )
            if not raw_text:
                raise ValueError(f"Gemini Interactions API ({model_name}) returned empty text output.")
            return raw_text, tokens_used
        except Exception as inter_err:
            if _is_rate_limit_error(inter_err):
                logger.warning(
                    "Gemini rate limit (429) on interactions.create (%s): %s. Skipping retry to avoid backoff delays.",
                    model_name,
                    inter_err,
                )
                raise RuntimeError(f"Gemini rate limit: {inter_err}") from inter_err

            logger.debug(
                "Interactions API note (%s): %s. Trying generate_content fallback.",
                model_name,
                inter_err,
            )
            # 2. Fallback to client.models.generate_content for the same model
            try:
                from google.genai import types

                response = client.models.generate_content(
                    model=model_name,
                    contents=[system_instruction, user_content],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema_model,
                        http_options=types.HttpOptions(
                            timeout=timeout_ms,
                            retry_options=types.HttpRetryOptions(attempts=1),
                        ),
                    ),
                )
                raw_text = response.text
                tokens_used = (
                    getattr(response.usage_metadata, "total_token_count", None)
                    if hasattr(response, "usage_metadata") and response.usage_metadata
                    else None
                )
                if not raw_text:
                    raise ValueError(f"Gemini generate_content ({model_name}) returned empty text output.")
                return raw_text, tokens_used
            except Exception as gen_err:
                if _is_rate_limit_error(gen_err):
                    raise RuntimeError(f"Gemini rate limit: {gen_err}") from gen_err
                logger.warning("Gemini generate_content failed for %s: %s", model_name, gen_err)
                raise RuntimeError(f"{model_name}: {inter_err}; {gen_err}") from gen_err

    def call_gemini_structured(
        self,
        system_instruction: str,
        user_content: str,
        schema_model: Any,
        timeout_ms: int = 30_000,
    ) -> Tuple[str, Optional[int]]:
        """Invoke Gemini with per-call timeout, fast-fail on 429, and model fallback on 503/504."""
        client = self.get_gemini_client(timeout_ms=timeout_ms)
        schema = schema_model.model_json_schema()
        candidate_models = list(dict.fromkeys([self.gemini_model, *self.gemini_fallback_models]))
        errors: List[str] = []

        raise_on_all_429: Optional[Exception] = None
        for idx, model_name in enumerate(candidate_models):
            try:
                raw_text, tokens_used = self._invoke_gemini_model(
                    client=client,
                    model_name=model_name,
                    system_instruction=system_instruction,
                    user_content=user_content,
                    schema=schema,
                    schema_model=schema_model,
                    timeout_ms=timeout_ms,
                )
                if idx > 0:
                    logger.info("✅ Gemini backup model '%s' succeeded.", model_name)
                self.last_gemini_model_used = model_name
                return raw_text, tokens_used
            except Exception as err:
                errors.append(str(err))
                has_next = idx + 1 < len(candidate_models)
                if _is_rate_limit_error(err):
                    raise_on_all_429 = err
                    if has_next:
                        logger.warning(
                            "Gemini model '%s' hit rate limit (429); failing over to backup '%s'.",
                            model_name,
                            candidate_models[idx + 1],
                        )
                        continue
                    raise
                raise_on_all_429 = None
                if has_next and _is_transient_gemini_error(err):
                    logger.warning(
                        "Gemini model '%s' hit transient overload/timeout; failing over to '%s'.",
                        model_name,
                        candidate_models[idx + 1],
                    )
                    continue
                if has_next:
                    logger.warning(
                        "Gemini model '%s' failed (%s); trying backup '%s'.",
                        model_name,
                        err,
                        candidate_models[idx + 1],
                    )
                    continue
                break

        if raise_on_all_429 is not None:
            raise raise_on_all_429
        raise RuntimeError(f"Gemini generation failed: {'; '.join(errors)}")

    def _call_gemini_interactions(
        self,
        article_text: str,
        meta: Dict[str, Any],
    ) -> Tuple[ArticleDigestOutput, str, Optional[int]]:
        """Call Gemini Interactions API with structured text schema and continuation support."""
        user_content = (
            f"Please synthesize the following article into a half-page, knowledge-rich digest.\n\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Source: {meta.get('source_name', 'Unknown')} ({meta.get('source_type', 'article')})\n"
            f"URL: {meta.get('url', '')}\n\n"
            f"--- CONTENT ---\n{article_text}"
        )

        raw_output_text, tokens_used = self.call_gemini_structured(
            system_instruction=SYSTEM_PROMPT,
            user_content=user_content,
            schema_model=ArticleDigestOutput,
            timeout_ms=30_000,
        )

        cleaned = clean_json_text(raw_output_text)
        data = json.loads(cleaned)
        output = ArticleDigestOutput.model_validate(data)
        used_model = self.last_gemini_model_used or self.gemini_model

        return output, f"gemini:interactions/{used_model}", tokens_used

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

