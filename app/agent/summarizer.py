"""AI News Digest Agent.

Coordinates:
- Fetching articles across all 3 sources (YouTube, OpenAI, Anthropic) from PostgreSQL
- Generating knowledge-rich, half-page structured summaries using OpenAI Responses API / Gemini Interactions API
- Saving results into the new 'digest' table
- Marking articles as summarized
- CLI interface for batch executions
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.llm_client import DualLLMClient
from app.agent.schemas import AgentRunReport, ArticleDigestOutput, DigestResult
from app.database.models import Article, ArticleDigest
from app.database.repository import NewsRepository
from app.database.session import SessionLocal

logger = logging.getLogger("ai_news.agent.summarizer")


class DigestAgent:
    """Agent that synthesizes ingested articles into half-page digests in table 'digest'."""

    def __init__(
        self,
        session: Optional[Session] = None,
        llm_client: Optional[DualLLMClient] = None,
        preferred_provider: str = "gemini",
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> None:
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.repo = NewsRepository(session=self.session)
        self.llm_client = llm_client or DualLLMClient(
            preferred_provider=preferred_provider,
            openai_model=openai_model,
            gemini_model=gemini_model,
        )

    def close(self) -> None:
        """Close session if owned."""
        if self._owns_session and self.session:
            self.session.close()

    def __enter__(self) -> DigestAgent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_articles_to_process(
        self,
        source_type: Optional[str] = None,
        unsummarized_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[Article]:
        """Fetch candidate articles from the database for summarization.

        Filters:
            source_type: Optional filter for 'youtube', 'openai', or 'anthropic'
            unsummarized_only: Only process articles where is_summarized is False
            limit: Maximum articles to retrieve
        """
        stmt = select(Article).order_by(Article.published_at.desc())

        if unsummarized_only:
            stmt = stmt.where(Article.is_summarized.is_(False))

        if source_type:
            from app.database.models import Source

            stmt = stmt.join(Article.source).where(Source.source_type == source_type)

        if limit:
            stmt = stmt.limit(limit)

        return list(self.session.execute(stmt).scalars().all())

    def process_article(
        self,
        article: Article,
        force: bool = False,
        dry_run: bool = False,
        provider_override: Optional[str] = None,
    ) -> DigestResult:
        """Process an individual article and persist its digest entry."""
        source_name = article.source.name if article.source else (article.category or "Unknown Source")
        source_type = article.source.source_type if article.source else "article"

        logger.info(
            "Processing article ID=%d: '%s' [%s] (%s)",
            article.id,
            article.title[:50],
            source_type,
            article.url,
        )

        # Check if already summarized in table 'digest'
        existing_digest = self.repo.article_digests.get_by_article_id(article.id)
        if existing_digest and not force:
            logger.info("Article ID=%d already has an entry in table 'digest'. Skipping (force=False).", article.id)
            return DigestResult(
                article_id=article.id,
                title=existing_digest.title,
                url=existing_digest.url,
                source_name=existing_digest.source_name,
                source_type=existing_digest.source_type,
                category=existing_digest.category,
                output=ArticleDigestOutput(
                    title=existing_digest.title,
                    summary=existing_digest.summary,
                    problem_statement=existing_digest.problem_statement or "",
                    solution_statement=existing_digest.solution_statement or "",
                    key_insights=existing_digest.key_insights or [],
                ),
                model_used=existing_digest.model_used,
                tokens_used=existing_digest.tokens_used,
                saved_to_db=True,
            )

        # Prepare formatted markdown for LLM input
        llm_input_text = article.to_llm_markdown()
        metadata = {
            "article_id": article.id,
            "title": article.title,
            "url": article.url,
            "source_name": source_name,
            "source_type": source_type,
            "category": article.category,
            "word_count": article.word_count,
        }

        # Generate structured digest
        output, model_used, tokens_used = self.llm_client.generate_digest(
            article_text=llm_input_text,
            metadata=metadata,
            provider_override=provider_override,
        )

        saved = False
        if not dry_run:
            # Persist to table 'digest'
            self.repo.article_digests.upsert(
                article_id=article.id,
                title=output.title or article.title,
                url=article.url,
                source_name=source_name,
                source_type=source_type,
                summary=output.summary,
                problem_statement=output.problem_statement,
                solution_statement=output.solution_statement,
                key_insights=output.key_insights,
                model_used=model_used,
                category=article.category,
                tokens_used=tokens_used,
            )
            # Update article record
            article.is_summarized = True
            self.session.commit()
            saved = True
            logger.info("✅ Saved digest entry for article ID=%d to table 'digest' (%s)", article.id, model_used)

        return DigestResult(
            article_id=article.id,
            title=output.title or article.title,
            url=article.url,
            source_name=source_name,
            source_type=source_type,
            category=article.category,
            output=output,
            model_used=model_used,
            tokens_used=tokens_used,
            saved_to_db=saved,
        )

    def run(
        self,
        source_type: Optional[str] = None,
        unsummarized_only: bool = True,
        limit: Optional[int] = 12,
        force: bool = False,
        dry_run: bool = False,
        provider_override: Optional[str] = None,
    ) -> AgentRunReport:
        """Run batch summarization over articles from database."""
        articles = self.get_articles_to_process(
            source_type=source_type,
            unsummarized_only=(unsummarized_only and not force),
            limit=limit,
        )

        report = AgentRunReport(total_articles=len(articles))
        logger.info("Found %d articles to summarize.", len(articles))

        for art in articles:
            try:
                result = self.process_article(
                    article=art,
                    force=force,
                    dry_run=dry_run,
                    provider_override=provider_override,
                )
                report.processed += 1
                report.successful += 1
                report.results.append(result)

                # Record model breakdown
                m = result.model_used
                report.model_breakdown[m] = report.model_breakdown.get(m, 0) + 1

                # Polite pause to stay well within Gemini free-tier rate limits
                import time
                time.sleep(1.0)

            except Exception as e:
                logger.error("Failed to process article ID=%d: %s", art.id, e, exc_info=True)
                report.processed += 1
                report.failed += 1
                report.results.append(
                    DigestResult(
                        article_id=art.id,
                        title=art.title,
                        url=art.url,
                        source_name=art.source.name if art.source else "Unknown",
                        source_type=art.source.source_type if art.source else "article",
                        category=art.category,
                        output=ArticleDigestOutput(
                            title=art.title,
                            summary="Failed to summarize.",
                            problem_statement="",
                            solution_statement="",
                            key_insights=[],
                        ),
                        model_used="none",
                        error=str(e),
                    )
                )

        return report


def main() -> None:
    """CLI runner for DigestAgent."""
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run AI News Digest Agent to summarize articles into table 'digest'",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["youtube", "openai", "anthropic"],
        default=None,
        help="Filter articles by source type",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of articles to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize articles that already have entries in table 'digest'",
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
        help="Generate summaries without writing to PostgreSQL table 'digest'",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed DEBUG logging",
    )

    args = parser.parse_args()

    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("🤖 AI NEWS DIGEST AGENT")
    print(f"Target Sources: YouTube, OpenAI, Anthropic | Provider Strategy: {args.provider}")
    print("=" * 70)

    with DigestAgent(preferred_provider=args.provider) as agent:
        report = agent.run(
            source_type=args.source,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 70)
    print("📊 AGENT EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Found:  {report.total_articles}")
    print(f"Processed:    {report.processed}")
    print(f"Successful:   {report.successful}")
    print(f"Failed:       {report.failed}")
    print("Model Breakdown:")
    for model_name, count in report.model_breakdown.items():
        print(f"  - {model_name}: {count}")

    if report.results:
        print("\n" + "-" * 70)
        print("📝 SAMPLE DIGEST ENTRY:")
        print("-" * 70)
        sample = report.results[0]
        print(f"Article ID:    {sample.article_id}")
        print(f"Title:         {sample.title}")
        print(f"Source:        {sample.source_name} ({sample.source_type})")
        print(f"Original URL:  {sample.url}")
        print(f"Model Used:    {sample.model_used}")
        print(f"\n--- Half-Page Summary ---\n{sample.output.summary}")
        print(f"\n--- Problem Statement ---\n{sample.output.problem_statement}")
        print(f"\n--- Solution Statement ---\n{sample.output.solution_statement}")
        print("\n--- Key Insights ---")
        for idx, insight in enumerate(sample.output.key_insights, 1):
            print(f"  {idx}. {insight}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

