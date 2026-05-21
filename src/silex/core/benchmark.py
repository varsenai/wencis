"""
Benchmark Suite — Phase 7.

A fixed evaluation suite that measures ARIA's performance across domains.
Runs a set of challenging cross-domain questions through the cognitive loop
and uses an LLM judge to score each response.
"""

from __future__ import annotations

import json
from typing import Callable, Any

from silex.llm.base import SupportsLLM
from silex.models.schemas import (
    BenchmarkQuestion,
    BenchmarkResult,
    CritiqueScore,
)
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.benchmark")

# Fixed benchmark questions — these never change so we can track progress
BENCHMARK_SUITE: list[BenchmarkQuestion] = [
    # Physics / Science
    BenchmarkQuestion(
        domain="physics",
        question="Explain why time dilation occurs near massive objects, and what this implies about the nature of spacetime itself.",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        domain="biology",
        question="How does CRISPR-Cas9 work at a molecular level, and what are the key ethical concerns about germline editing?",
        difficulty="medium",
    ),
    # Economics / Systems
    BenchmarkQuestion(
        domain="economics",
        question="Explain the mechanism by which quantitative easing can simultaneously inflate asset prices while consumer inflation remains low.",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        domain="game_theory",
        question="What is the Nash Equilibrium and why does it sometimes produce outcomes that are worse for all players? Give a real-world example.",
        difficulty="medium",
    ),
    # Philosophy / Reasoning
    BenchmarkQuestion(
        domain="philosophy",
        question="Is the Ship of Theseus paradox a genuine metaphysical puzzle, or a confusion about language? Argue both sides.",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        domain="epistemology",
        question="What is the Gettier problem and why does it matter for artificial intelligence systems that claim to 'know' things?",
        difficulty="medium",
    ),
    # Cross-domain Transfer
    BenchmarkQuestion(
        domain="cross_domain",
        question="What structural principle is shared by both the immune system's response to novel pathogens and a startup's pivot strategy when entering a new market?",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        domain="cross_domain",
        question="How is the concept of 'entropy' from thermodynamics analogous to information theory, and what does this tell us about the nature of disorder?",
        difficulty="hard",
    ),
]

JUDGE_PROMPT = """You are a strict, impartial judge evaluating an AI system's response.
Score the response on three dimensions, each from 0.0 to 1.0:

1. accuracy: Is the information factually correct? Are there errors or hallucinations?
2. depth: Does the response show genuine understanding, or is it superficial?
3. honesty: Does the response admit uncertainty where appropriate?

Be harsh. A mediocre response should score around 0.5.
A genuinely excellent response scores 0.8+.
Reserve 0.9+ for responses that demonstrate real intellectual depth."""


class BenchmarkRunner:
    """Runs the fixed benchmark suite and tracks scores over time."""

    def __init__(self, llm_client: SupportsLLM, db: Database):
        self.llm = llm_client
        self.db = db

    async def run(
        self,
        cognitive_loop,
        status_callback: Callable[..., Any] | None = None,
    ) -> BenchmarkResult:
        """
        Run the full benchmark suite.
        Returns a BenchmarkResult with aggregate scores.
        """
        scores: list[CritiqueScore] = []
        domains_seen = set()

        for i, q in enumerate(BENCHMARK_SUITE):
            if status_callback:
                status_callback(
                    f"[bright_magenta]  Benchmark [{i+1}/{len(BENCHMARK_SUITE)}]: "
                    f"{q.domain} ({q.difficulty})...[/]"
                )

            try:
                # Get ARIA's response
                cognitive = await cognitive_loop.process(
                    q.question, status_callback=None
                )

                # Judge the response
                score = await self._judge_response(q, cognitive.response)
                scores.append(score)
                domains_seen.add(q.domain)

                if status_callback:
                    status_callback(
                        f"[dim]    Acc={score.accuracy:.2f} Dep={score.depth:.2f} "
                        f"Hon={score.honesty:.2f}[/]"
                    )

            except Exception as e:
                log.warning(f"Benchmark question failed: {e}")
                # Score a failure as 0
                scores.append(CritiqueScore(accuracy=0.0, depth=0.0, honesty=0.0))
                domains_seen.add(q.domain)

        # Aggregate
        if not scores:
            return BenchmarkResult(
                total_score=0.0,
                accuracy_avg=0.0,
                depth_avg=0.0,
                honesty_avg=0.0,
                domains_tested=list(domains_seen),
                question_count=0,
            )

        acc_avg = sum(s.accuracy for s in scores) / len(scores)
        dep_avg = sum(s.depth for s in scores) / len(scores)
        hon_avg = sum(s.honesty for s in scores) / len(scores)
        total = ((acc_avg + dep_avg + hon_avg) / 3.0) * 100.0

        result = BenchmarkResult(
            total_score=round(total, 2),
            accuracy_avg=round(acc_avg, 3),
            depth_avg=round(dep_avg, 3),
            honesty_avg=round(hon_avg, 3),
            domains_tested=sorted(domains_seen),
            question_count=len(scores),
        )

        # Persist
        await self.db.execute(
            """
            INSERT INTO benchmark_history (
                id, total_score, accuracy_avg, depth_avg, honesty_avg,
                domains_tested_json, question_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                result.total_score,
                result.accuracy_avg,
                result.depth_avg,
                result.honesty_avg,
                json.dumps(result.domains_tested),
                result.question_count,
                result.created_at,
            ),
        )

        log.info(f"Benchmark complete: {result.total_score:.1f}/100")
        return result

    async def get_history(self) -> list[BenchmarkResult]:
        """Fetch historical benchmark results."""
        rows = await self.db.fetch_all(
            "SELECT * FROM benchmark_history ORDER BY created_at DESC"
        )
        return [
            BenchmarkResult(
                id=r["id"],
                total_score=r["total_score"],
                accuracy_avg=r["accuracy_avg"],
                depth_avg=r["depth_avg"],
                honesty_avg=r["honesty_avg"],
                domains_tested=json.loads(r["domains_tested_json"]),
                question_count=r["question_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def _judge_response(
        self, question: BenchmarkQuestion, response: str
    ) -> CritiqueScore:
        """Use an LLM judge to score a response."""
        content = (
            f"DOMAIN: {question.domain}\n"
            f"DIFFICULTY: {question.difficulty}\n"
            f"QUESTION: {question.question}\n\n"
            f"ARIA'S RESPONSE:\n{response}\n\n"
            f"Score this response."
        )

        try:
            return await self.llm.complete_json(
                schema=CritiqueScore,
                system_prompt=JUDGE_PROMPT,
                user_input=content,
                temperature=0.1,
                request_kind="benchmark_judge",
            )

        except Exception as e:
            log.warning(f"Judge failed: {e}")
            return CritiqueScore(accuracy=0.0, depth=0.0, honesty=0.0)
