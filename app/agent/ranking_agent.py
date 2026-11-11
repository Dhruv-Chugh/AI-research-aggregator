"""Personalized User Interest Ranking Agent.

Ranks news digests from table 'digest' based on individual subscriber profiles,
technical interests, expertise level, and custom prioritization instructions.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.agent.llm_client import DualLLMClient, clean_json_text, estimate_tokens
from app.agent.ranking_schemas import (
    CandidateDigestInput,
    PersonalizedRankingOutput,
    RankedDigestItem,
    UserProfileData,
    UserRankingBatchReport,
)
from app.database.models import ArticleDigest, Subscriber, UserDigestRanking, utc_now
from app.database.repository import NewsRepository
from app.database.session import SessionLocal

logger = logging.getLogger("ai_news.agent.ranking")

RANKING_SYSTEM_PROMPT = """You are an elite Personalized AI Research Curator and Intelligence Officer.
Your objective is to evaluate, score, and rank a collection of AI news summaries specifically for a target user profile.

You will receive:
1. The User's Profile (Role/Bio, Expertise Level, Specific Topic Interests, Preferred Sources, and Prioritization Rules).
2. A list of candidate article and podcast summaries from table 'digest'.

Your instructions:
1. Deeply analyze how each candidate summary aligns with the user's specific technical interests, domain role, and expertise.
2. Respect the user's custom ranking instructions (e.g. prioritizing frontier research over corporate acquisitions).
3. Assign a precise 'rank_score' between 0.0 and 100.0 reflecting its personalized value to this user:
   - 90.0 - 100.0: Critical must-read directly matching core interests & role.
   - 75.0 - 89.9: High value, relevant technical or ecosystem development.
   - 50.0 - 74.9: Moderately relevant general AI news.
   - 0.0 - 49.9: Low relevance or outside stated focus.
4. Rank the items in descending order of priority, assigning 'rank_position' from 1 (top recommendation) to N.
5. Provide a personalized 2-3 sentence 'relevance_reasoning' specifically addressing why this matters for the user.
6. Provide 2-4 'personalized_takeaways' highlighting concrete implications for the user's domain.
7. Provide a 2-3 sentence 'executive_summary' overview of the day's briefing.

Return ONLY raw valid JSON conforming to the schema."""


def extract_quick_summary(
    summary_text: str,
    max_sentences: int = 3,
    max_chars: int = 380,
) -> str:
    """Extract a clean 3-4 line concise summary from a half-page digest text.

    Splits text on sentence boundaries, extracts up to max_sentences, and bounds
    length appropriately so it formats neatly in 3-4 lines for daily email briefings.
    """
    if not summary_text:
        return ""

    cleaned = " ".join(summary_text.strip().split())
    import re

    # Match sentences ending with period, exclamation, or question mark followed by whitespace
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    selected = sentences[:max_sentences]
    quick = " ".join(selected).strip()

    if len(quick) > max_chars:
        truncated = quick[:max_chars].rsplit(" ", 1)[0]
        if not truncated.endswith((".", "!", "?")):
            truncated += "..."
        quick = truncated

    return quick


def classify_source_diversity_tiers(
    ranked_items: List[RankedDigestItem],
    max_featured_per_source: int = 2,
    digest_map: Optional[Dict[int, CandidateDigestInput]] = None,
) -> List[RankedDigestItem]:
    """Classify ranked items into Two Tiers for the daily intelligence briefing:
    - Tier 1 ('featured'): Top N (default 2) items per source receive deep half-page coverage.
    - Tier 2 ('quick_bite'): All remaining items receive a compact 3-4 line concise summary.

    Preserves ALL items in the digest with contiguous global rank positions (1..N).
    No items are removed from the digest.
    """
    source_featured_counts: Dict[str, int] = {}

    for pos, item in enumerate(ranked_items, 1):
        item.rank_position = pos
        src = (item.source_type or "unknown").lower().strip()
        current_featured = source_featured_counts.get(src, 0)

        # Attach 3-4 line quick_summary if available
        if not item.quick_summary:
            if digest_map and item.digest_id in digest_map:
                item.quick_summary = extract_quick_summary(digest_map[item.digest_id].summary)
            elif item.relevance_reasoning:
                item.quick_summary = extract_quick_summary(item.relevance_reasoning)

        if max_featured_per_source and max_featured_per_source > 0:
            if current_featured < max_featured_per_source:
                item.tier = "featured"
                source_featured_counts[src] = current_featured + 1
            else:
                item.tier = "quick_bite"
        else:
            item.tier = "featured"

    return ranked_items


def apply_source_diversity_limit(
    ranked_items: List[RankedDigestItem],
    max_per_source: int = 2,
) -> List[RankedDigestItem]:
    """Filter ranked items to include at most max_per_source items per source type.

    Preserves descending ranking score order while ensuring balanced representation across
    sources (e.g. maximum 2 from YouTube, maximum 2 from OpenAI, maximum 2 from Anthropic).
    If a source has fewer than max_per_source items (e.g. 1), that item is kept.
    """
    if max_per_source is None or max_per_source <= 0:
        return ranked_items

    source_counts: Dict[str, int] = {}
    balanced: List[RankedDigestItem] = []

    for item in ranked_items:
        src = (item.source_type or "unknown").lower().strip()
        current_count = source_counts.get(src, 0)
        if current_count < max_per_source:
            source_counts[src] = current_count + 1
            balanced.append(item)

    # Re-index rank_position contiguously: 1, 2, 3...
    for pos, item in enumerate(balanced, 1):
        item.rank_position = pos

    return balanced


class UserRankingAgent:
    """Agent that ranks news digests according to user interest profiles."""

    def __init__(
        self,
        session: Optional[Session] = None,
        llm_client: Optional[DualLLMClient] = None,
        preferred_provider: str = "gemini",
    ) -> None:
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.repo = NewsRepository(session=self.session)
        self.llm_client = llm_client or DualLLMClient(preferred_provider=preferred_provider)

    def close(self) -> None:
        """Close session if owned."""
        if self._owns_session and self.session:
            self.session.close()

    def __enter__(self) -> UserRankingAgent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_candidate_digests(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[CandidateDigestInput]:
        """Fetch candidate digests from table 'digest' within lookback window.

        Falls back to all available digests if few or no digests exist strictly within the window.
        """
        cutoff = utc_now() - timedelta(hours=hours)
        stmt = (
            select(ArticleDigest)
            .where(ArticleDigest.created_at >= cutoff)
            .order_by(desc(ArticleDigest.created_at))
            .limit(limit)
        )
        records = list(self.session.execute(stmt).scalars().all())

        # Fallback / augment with recent digests if fewer than 3 found in exact window (e.g. testing or quiet news days)
        if len(records) < 3:
            logger.info("Only %d digests created in last %d hours. Fetching recent digests...", len(records), hours)
            stmt = select(ArticleDigest).order_by(desc(ArticleDigest.created_at)).limit(limit)
            records = list(self.session.execute(stmt).scalars().all())

        candidates: List[CandidateDigestInput] = []
        for r in records:
            candidates.append(
                CandidateDigestInput(
                    digest_id=r.id,
                    article_id=r.article_id,
                    title=r.title,
                    url=r.url,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    category=r.category,
                    summary=r.summary,
                    problem_statement=r.problem_statement,
                    solution_statement=r.solution_statement,
                    key_insights=r.key_insights or [],
                    model_used=r.model_used,
                    tokens_used=r.tokens_used,
                )
            )
        return candidates

    def get_subscriber_profiles(self, email: Optional[str] = None) -> List[UserProfileData]:
        """Fetch active subscribers and convert to UserProfileData."""
        if email:
            sub = self.repo.subscribers.get_by_email(email)
            subs = [sub] if sub else []
        else:
            subs = self.repo.subscribers.get_active()

        profiles: List[UserProfileData] = []
        for s in subs:
            interests = s.interests or []
            # Provide sensible defaults if user profile has not customized interests yet
            if not interests:
                interests = [
                    "Artificial Intelligence",
                    "Large Language Models & Reasoning",
                    "Frontier AI Research & Architecture",
                    "AI Agents & Autonomous Systems",
                    "Industry Strategy & Tech Leadership",
                ]

            preferred = s.preferred_sources or ["youtube", "openai", "anthropic"]
            instructions = s.ranking_instructions or (
                "Prioritize breakthrough technical capabilities, scientific advances, "
                "and high-signal strategic interviews over basic business announcements."
            )

            profiles.append(
                UserProfileData(
                    subscriber_id=s.id,
                    email=s.email,
                    name=s.name or s.email.split("@")[0],
                    bio_or_role=s.bio_or_role or "AI Engineer & Practitioner",
                    expertise_level=s.expertise_level or "intermediate",
                    interests=interests,
                    preferred_sources=preferred,
                    ranking_instructions=instructions,
                )
            )
        return profiles

    def rank_for_user(
        self,
        user: UserProfileData,
        candidates: List[CandidateDigestInput],
        batch_date: Optional[date] = None,
        max_per_source: int = 2,
    ) -> PersonalizedRankingOutput:
        """Evaluate, score, and rank candidate digests for a user profile.

        Args:
            user: User interest profile and preferences.
            candidates: Candidate summaries from table 'digest'.
            batch_date: Date for the batch execution.
            max_per_source: Maximum items allowed from any single source (e.g. max 2 from YouTube,
                            OpenAI, Anthropic). If a source has <= max_per_source, all are kept.
        """
        b_date = batch_date or utc_now().date()

        if not candidates:
            return PersonalizedRankingOutput(
                executive_summary="No news digests were available in this 24-hour cycle.",
                ranked_items=[],
            )

        # Format candidates payload for LLM
        items_payload = []
        for c in candidates:
            items_payload.append({
                "digest_id": c.digest_id,
                "title": c.title,
                "source": f"{c.source_name} ({c.source_type})",
                "source_type": c.source_type,
                "url": c.url,
                "summary": c.summary,
                "problem_statement": c.problem_statement or "",
                "solution_statement": c.solution_statement or "",
                "key_insights": c.key_insights,
            })

        user_prompt = (
            f"=== TARGET USER PROFILE ===\n"
            f"Name: {user.name}\n"
            f"Role/Bio: {user.bio_or_role}\n"
            f"Expertise Level: {user.expertise_level}\n"
            f"Stated Interests: {', '.join(user.interests)}\n"
            f"Preferred Sources: {', '.join(user.preferred_sources)}\n"
            f"Custom Instructions: {user.ranking_instructions}\n\n"
            f"=== CANDIDATE DIGESTS TO RANK ({len(candidates)} items) ===\n"
            f"{json.dumps(items_payload, indent=2)}\n\n"
            f"Evaluate, score (0-100), rank, and generate tailored insights according to the schema."
        )

        try:
            ranking_output = self._call_llm_ranking(user_prompt)
            # Reconcile URLs, titles, and source types
            digest_map = {c.digest_id: c for c in candidates}
            for item in ranking_output.ranked_items:
                if item.digest_id in digest_map:
                    cand = digest_map[item.digest_id]
                    item.url = cand.url
                    item.title = cand.title
                    item.source_name = cand.source_name
                    item.source_type = cand.source_type

            # Classify into Two Tiers: Featured (max_per_source) + In Brief / Quick Bites (remaining)
            ranking_output.ranked_items = classify_source_diversity_tiers(
                ranking_output.ranked_items,
                max_featured_per_source=max_per_source,
                digest_map=digest_map,
            )

            return ranking_output
        except Exception as e:
            logger.warning("Live LLM ranking failed: %s. Using deterministic profile matcher.", e)
            return self._deterministic_profile_matcher(user, candidates, max_per_source=max_per_source)

    def _call_llm_ranking(self, prompt: str) -> PersonalizedRankingOutput:
        """Invoke DualLLMClient with structured JSON schema for ranking."""
        schema = PersonalizedRankingOutput.model_json_schema()
        raw_text: Optional[str] = None

        # 1. Try OpenAI Responses API first if key configured
        if self.llm_client.has_openai_key:
            try:
                client = self.llm_client.get_openai_client()
                response = client.responses.create(
                    model=self.llm_client.openai_model,
                    instructions=RANKING_SYSTEM_PROMPT,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "personalized_ranking",
                            "schema": schema,
                        }
                    },
                )
                raw_text = response.output_text
            except Exception as e:
                logger.warning("OpenAI ranking call error: %s. Falling back to Gemini...", e)

        # 2. Try Gemini Interactions API if OpenAI was unavailable or failed (45s timeout for batch ranking)
        if not raw_text and self.llm_client.has_gemini_key:
            try:
                raw_text, _ = self.llm_client.call_gemini_structured(
                    system_instruction=RANKING_SYSTEM_PROMPT,
                    user_content=prompt,
                    schema_model=PersonalizedRankingOutput,
                    timeout_ms=45_000,
                )
            except Exception as e:
                logger.warning("Gemini ranking call error: %s.", e)

        if not raw_text:
            raise RuntimeError("No LLM response text generated for ranking.")

        cleaned = clean_json_text(raw_text)
        data = json.loads(cleaned)
        return PersonalizedRankingOutput.model_validate(data)

    def _deterministic_profile_matcher(
        self,
        user: UserProfileData,
        candidates: List[CandidateDigestInput],
        max_per_source: int = 2,
    ) -> PersonalizedRankingOutput:
        """Deterministic keyword and heuristic scoring when LLM APIs are offline or mock testing."""
        user_interest_keywords = set()
        for interest in user.interests:
            for word in re.findall(r"\w+", interest.lower()):
                if len(word) > 3:
                    user_interest_keywords.add(word)

        scored_items: List[Tuple[float, CandidateDigestInput, str, List[str]]] = []

        for c in candidates:
            text_corpus = f"{c.title} {c.summary} {c.problem_statement or ''} {c.solution_statement or ''}".lower()
            corpus_words = set(re.findall(r"\w+", text_corpus))

            # Base score
            overlap = len(user_interest_keywords.intersection(corpus_words))
            base_score = 55.0 + min(overlap * 8.0, 35.0)

            # Preferred source bonus
            if c.source_type.lower() in [s.lower() for s in user.preferred_sources]:
                base_score += 7.0

            # Source / content richness bonus
            if c.source_type == "youtube":
                base_score += 3.0  # long-form discussions
            elif "research" in (c.category or "").lower() or "research" in c.title.lower():
                base_score += 4.0

            final_score = min(round(base_score, 1), 99.0)

            reasoning = (
                f"Ranked for {user.name} ({user.bio_or_role}) due to strong relevance to "
                f"topics in {c.source_name} (`{c.source_type}`). "
                f"Addresses core interests with {overlap} direct domain keyword matches."
            )

            takeaways = [
                f"Key relevance for {user.bio_or_role}: directly intersects with {c.title}.",
                f"Practical takeaway: examine the solution methodology outlined in {c.source_name}.",
                f"Strategic takeaway: track follow-up developments from {c.url}.",
            ]

            scored_items.append((final_score, c, reasoning, takeaways))

        # Sort descending by score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        ranked: List[RankedDigestItem] = []
        for pos, (score, c, reasoning, takeaways) in enumerate(scored_items, 1):
            ranked.append(
                RankedDigestItem(
                    digest_id=c.digest_id,
                    title=c.title,
                    url=c.url,
                    rank_position=pos,
                    rank_score=score,
                    relevance_reasoning=reasoning,
                    personalized_takeaways=takeaways,
                    source_name=c.source_name,
                    source_type=c.source_type,
                )
            )

        # Classify into Two Tiers: Featured (max_per_source) + In Brief / Quick Bites (remaining)
        digest_map = {c.digest_id: c for c in candidates}
        ranked = classify_source_diversity_tiers(
            ranked,
            max_featured_per_source=max_per_source,
            digest_map=digest_map,
        )

        top_item = ranked[0].title if ranked else "AI ecosystem updates"
        exec_summary = (
            f"Daily personalized intelligence briefing for {user.name} ({user.bio_or_role}). "
            f"The primary focus centers on '{top_item}', aligning closely with your stated interests in "
            f"{', '.join(user.interests[:3])}."
        )

        return PersonalizedRankingOutput(
            executive_summary=exec_summary,
            ranked_items=ranked,
        )

    def persist_rankings(
        self,
        subscriber_id: int,
        ranked_items: List[RankedDigestItem],
        batch_date: date,
    ) -> int:
        """Persist ranked items to table 'user_digest_rankings', clearing previous batch for date."""
        self.repo.user_rankings.clear_subscriber_batch(subscriber_id, batch_date)

        saved_count = 0
        for item in ranked_items:
            self.repo.user_rankings.create(
                subscriber_id=subscriber_id,
                digest_id=item.digest_id,
                rank_position=item.rank_position,
                rank_score=item.rank_score,
                relevance_reasoning=item.relevance_reasoning,
                key_takeaways_for_user=item.personalized_takeaways,
                batch_date=batch_date,
                tier=item.tier,
                quick_summary=item.quick_summary,
            )
            saved_count += 1

        self.session.commit()
        return saved_count

    def run_daily_ranking(
        self,
        hours: int = 24,
        email_filter: Optional[str] = None,
        dry_run: bool = False,
        max_per_source: int = 2,
    ) -> UserRankingBatchReport:
        """Execute a complete 24-hour personalized ranking cycle across active subscribers.

        Args:
            hours: Lookback window in hours for candidate digests.
            email_filter: Optional filter to a single subscriber.
            dry_run: If True, compute rankings without saving to PostgreSQL.
            max_per_source: Maximum items allowed per source (e.g. max 2 from YouTube, OpenAI, Anthropic).
        """
        batch_date = utc_now().date()
        report = UserRankingBatchReport(
            batch_date=batch_date.isoformat(),
            executed_at=utc_now().isoformat(),
        )

        candidates = self.get_candidate_digests(hours=hours)
        report.total_digests_available = len(candidates)
        logger.info("Found %d candidate digests from table 'digest'.", len(candidates))

        if not candidates:
            logger.warning("No candidate digests found in table 'digest'. Run summarizer first.")
            return report

        subscribers = self.get_subscriber_profiles(email=email_filter)
        report.subscribers_evaluated = len(subscribers)
        logger.info("Ranking digests for %d subscriber profile(s)...", len(subscribers))

        for sub in subscribers:
            try:
                ranking_result = self.rank_for_user(
                    sub, candidates, batch_date=batch_date, max_per_source=max_per_source
                )

                saved_count = 0
                if not dry_run:
                    saved_count = self.persist_rankings(
                        subscriber_id=sub.subscriber_id,
                        ranked_items=ranking_result.ranked_items,
                        batch_date=batch_date,
                    )
                    report.rankings_stored += saved_count

                report.rankings_by_subscriber[sub.email] = {
                    "name": sub.name,
                    "role": sub.bio_or_role,
                    "interests": sub.interests,
                    "executive_summary": ranking_result.executive_summary,
                    "ranked_count": len(ranking_result.ranked_items),
                    "featured_count": sum(1 for it in ranking_result.ranked_items if it.tier == "featured"),
                    "quick_bites_count": sum(1 for it in ranking_result.ranked_items if it.tier == "quick_bite"),
                    "top_item": (
                        ranking_result.ranked_items[0].model_dump()
                        if ranking_result.ranked_items
                        else None
                    ),
                    "items": [it.model_dump() for it in ranking_result.ranked_items],
                }

                logger.info(
                    "✅ Ranked %d items for %s (%s). Persisted: %d",
                    len(ranking_result.ranked_items),
                    sub.email,
                    sub.bio_or_role,
                    saved_count,
                )

            except Exception as e:
                err = f"Failed to rank for subscriber {sub.email}: {e}"
                logger.error(err, exc_info=True)
                report.errors.append(err)

        return report

    def export_personalized_markdown(
        self,
        email: str,
        batch_date: Optional[date] = None,
    ) -> str:
        """Generate a clean Markdown intelligence briefing for a subscriber."""
        b_date = batch_date or utc_now().date()
        sub = self.repo.subscribers.get_by_email(email)
        if not sub:
            return f"# Error: Subscriber '{email}' not found."

        rankings = self.repo.user_rankings.get_rankings_for_subscriber(sub.id, batch_date=b_date)
        if not rankings:
            return f"# No rankings found for '{email}' on {b_date}."

        lines = [
            f"# 🎯 Daily Personalized AI Intelligence Briefing",
            f"- **User**: {sub.name or sub.email} (`{sub.bio_or_role or 'AI Practitioner'}`)",
            f"- **Date**: {b_date.strftime('%B %d, %Y')}",
            f"- **Expertise Level**: `{sub.expertise_level}`",
            f"- **Focus Areas**: {', '.join(sub.interests or ['General AI'])}",
            "",
            "---",
            "",
        ]

        featured = [r for r in rankings if getattr(r, "tier", "featured") == "featured"]
        quick_bites = [r for r in rankings if getattr(r, "tier", "featured") == "quick_bite"]

        if not featured and not quick_bites:
            featured = rankings

        # Section 1: Featured Deep Dives (max 2 per source)
        if featured:
            lines.extend([
                "# 🌟 Featured Deep Dives",
                "> In-depth strategic and technical breakdowns tailored to your focus areas (max 2 per source).",
                "",
            ])
            for r in featured:
                digest = r.digest
                score_bar = "🔥 Top Recommendation" if r.rank_position == 1 else f"Rank #{r.rank_position}"
                lines.extend([
                    f"## {r.rank_position}. {digest.title} [{r.rank_score:.1f}/100 - {score_bar}]",
                    f"- **Source**: {digest.source_name} (`{digest.source_type}`)",
                    f"- **Original URL**: [{digest.url}]({digest.url})",
                    "",
                    f"### 💡 Why This Matters For You",
                    f"> {r.relevance_reasoning}",
                    "",
                    f"### 📖 Summary",
                    f"{digest.summary.strip()}",
                    "",
                ])

                if digest.problem_statement:
                    lines.extend([
                        f"**Problem Highlighted**: {digest.problem_statement.strip()}",
                        "",
                    ])

                if digest.solution_statement:
                    lines.extend([
                        f"**Solution Proposed**: {digest.solution_statement.strip()}",
                        "",
                    ])

                takeaways = r.key_takeaways_for_user or digest.key_insights or []
                if takeaways:
                    lines.append("### 🎯 Personalized Takeaways For Your Role")
                    for t in takeaways:
                        lines.append(f"- {t}")
                    lines.append("")

                lines.extend(["---", ""])

        # Section 2: In Brief & Ecosystem Radar (3-4 lines each)
        if quick_bites:
            lines.extend([
                "# ⚡ In Brief & Ecosystem Radar (Quick Takes)",
                "> Concise 3-4 line summaries of additional developments across OpenAI, Anthropic, and YouTube to keep you fully informed without the clutter.",
                "",
            ])
            for r in quick_bites:
                digest = r.digest
                brief_text = r.quick_summary or extract_quick_summary(digest.summary)
                lines.extend([
                    f"### {r.rank_position}. [{digest.title}]({digest.url}) [{r.rank_score:.1f}/100]",
                    f"- **Source**: {digest.source_name} (`{digest.source_type}`) | **Original Link**: [{digest.url}]({digest.url})",
                    f"- **Quick Brief (3-4 lines)**: {brief_text}",
                    f"- **Role Relevance**: *{r.relevance_reasoning}*",
                    "",
                ])
            lines.extend(["---", ""])

        return "\n".join(lines)


def main() -> None:
    """CLI runner for UserRankingAgent."""
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run Personalized User Interest Ranking Agent (every 24 hrs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        default=True,
        help="Execute 24-hour personalized ranking cycle immediately",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Filter to a specific subscriber email",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours for candidate digests",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "auto"],
        default="gemini",
        help="LLM provider strategy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and rank without saving to PostgreSQL 'user_digest_rankings' table",
    )
    parser.add_argument(
        "--export-md",
        action="store_true",
        help="Print the personalized Markdown briefing to stdout",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Start the background 24-hour scheduler daemon using APScheduler",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Interval hours when running in daemon mode",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=2,
        help="Maximum top items to include per source type (e.g. max 2 from YouTube, Anthropic, OpenAI)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.daemon:
        from app.agent.ranking_scheduler import start_ranking_scheduler

        print("\n" + "=" * 70)
        print("🕒 STARTING 24-HOUR PERSONALIZED RANKING SCHEDULER DAEMON")
        print(f"Interval: Every {args.interval_hours} hours | Provider: {args.provider}")
        print("=" * 70 + "\n")
        start_ranking_scheduler(interval_hours=args.interval_hours, provider=args.provider)
        return

    print("\n" + "=" * 70)
    print("🎯 PERSONALIZED USER INTEREST RANKING AGENT")
    print(f"Cycle: 24-Hour Batch | Provider: {args.provider}")
    print("=" * 70)

    with UserRankingAgent(preferred_provider=args.provider) as agent:
        report = agent.run_daily_ranking(
            hours=args.hours,
            email_filter=args.email,
            dry_run=args.dry_run,
            max_per_source=args.max_per_source,
        )

        print("\n" + "=" * 70)
        print("📊 24-HOUR RANKING EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Batch Date:               {report.batch_date}")
        print(f"Subscribers Evaluated:    {report.subscribers_evaluated}")
        print(f"Digests Available:        {report.total_digests_available}")
        print(f"Rankings Persisted:       {report.rankings_stored}")
        if report.errors:
            print("Errors Encountered:")
            for e in report.errors:
                print(f"  ❌ {e}")

        for email, data in report.rankings_by_subscriber.items():
            print("\n" + "-" * 70)
            print(f"👤 Briefing for: {data['name']} ({email})")
            print(f"Role: {data['role']}")
            print(f"Executive Summary: {data['executive_summary']}")
            print(
                f"Ranked Items ({data['ranked_count']} total: "
                f"{data.get('featured_count', 0)} featured, {data.get('quick_bites_count', 0)} in brief):"
            )
            for item in data.get("items", []):
                tier_badge = "🌟 FEATURED" if item.get("tier") == "featured" else "⚡ IN BRIEF"
                print(
                    f"  [{item['rank_position']}] [{tier_badge}] Score: {item['rank_score']:.1f}/100 | "
                    f"{item['title'][:40]}... | {item['url']}"
                )
                if item.get("tier") == "quick_bite" and item.get("quick_summary"):
                    print(f"      ⚡ Brief (3-4 lines): {item['quick_summary']}")
                else:
                    print(f"      Reason: {item['relevance_reasoning']}")

            if args.export_md:
                print("\n" + "=" * 70)
                print("📄 EXPORTED MARKDOWN BRIEFING")
                print("=" * 70)
                md = agent.export_personalized_markdown(email)
                print(md)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

