"""
Multi-Agent Debate Engine (Phase 4).

Orchestrates adversarial reasoning by splitting ARIA into three personas:
Agent A (Pro), Agent B (Con), and Judge (Synthesizer).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable, Any

from silex.llm.base import SupportsLLM
from silex.models.schemas import DebateArgument, DebateResolution, UncertaintyTopic
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.debate")

AGENT_A_PROMPT = """You are Agent A, a rigorous debater.
You have been assigned to argue the PRO or PERSPECTIVE 1 side of the following topic.
You must be logically rigorous, cite structural or causal reasons, and brutally deconstruct the opposing view.
Do NOT compromise. Your goal is to win the argument based on pure logic and evidence.
Be concise but devastating."""

AGENT_B_PROMPT = """You are Agent B, a rigorous debater.
You have been assigned to argue the CON or PERSPECTIVE 2 side of the following topic.
You must be logically rigorous, cite structural or causal reasons, and brutally deconstruct the opposing view.
Do NOT compromise. Your goal is to win the argument based on pure logic and evidence.
Be concise but devastating."""

JUDGE_PROMPT = """You are the Judge, the ultimate synthesizer of truth.
You are reviewing a debate between Agent A and Agent B.
Your job is NOT to pick a winner, but to find the truth. 
1. Identify the strongest, most structurally sound points from both sides.
2. Identify logical fallacies or weak assumptions made by either side.
3. Synthesize a final, nuanced truth that transcends the binary argument.
4. If the debate reveals new causal facts (e.g., A causes B, or X enables Y), extract them as graph updates.
You are objective, emotionless, and deeply wise."""

class DebateEngine:
    """Manages multi-agent debates and truth synthesis."""

    def __init__(self, llm_client: SupportsLLM, db: Database):
        self.llm = llm_client
        self.db = db

    async def run_debate(
        self,
        topic: str,
        rounds: int = 1,
        status_callback: Callable[..., Any] | None = None
    ) -> DebateResolution:
        """
        Run a full debate between A and B, judged by the Synthesizer.
        """
        log.info(f"Starting debate on topic: {topic}")
        transcript = []
        
        # Round 1
        if status_callback:
            status_callback("[red]  Agent A is formulating opening statement...[/]")
            
        a_arg = await self._generate_argument("Agent A", topic, transcript)
        transcript.append(a_arg)
        
        if status_callback:
            status_callback("[blue]  Agent B is rebutting...[/]")
            
        b_arg = await self._generate_argument("Agent B", topic, transcript)
        transcript.append(b_arg)
        
        # Additional rounds if requested
        for i in range(1, rounds):
            if status_callback:
                status_callback(f"[red]  Agent A is rebutting (Round {i+1})...[/]")
            a_arg = await self._generate_argument("Agent A", topic, transcript)
            transcript.append(a_arg)
            
            if status_callback:
                status_callback(f"[blue]  Agent B is rebutting (Round {i+1})...[/]")
            b_arg = await self._generate_argument("Agent B", topic, transcript)
            transcript.append(b_arg)
            
        # Judgment
        if status_callback:
            status_callback("[green]  The Judge is synthesizing the truth...[/]")
            
        resolution = await self._judge_debate(topic, transcript)
        
        # Save to DB
        await self._save_debate(topic, transcript, resolution)
        
        return resolution

    async def _generate_argument(
        self, agent_id: str, topic: str, transcript: list[DebateArgument]
    ) -> DebateArgument:
        """Call Gemini to generate a single debate turn."""
        prompt = AGENT_A_PROMPT if agent_id == "Agent A" else AGENT_B_PROMPT
        
        history = "TRANSCRIPT SO FAR:\n"
        for arg in transcript:
            history += f"{arg.agent_id}: {arg.claim}\nReasoning: {arg.reasoning}\n\n"
            
        content = f"TOPIC: {topic}\n\n{history}\nIt is your turn. Deliver your argument."
        
        data = (
            await self.llm.complete_json(
                schema=DebateArgument,
                system_prompt=prompt,
                user_input=content,
                temperature=0.7,
                request_kind="debate_argument",
            )
        ).model_dump()
        # Ensure the agent_id is correct regardless of what the model hallucinates
        data["agent_id"] = agent_id
        return DebateArgument(**data)

    async def _judge_debate(
        self, topic: str, transcript: list[DebateArgument]
    ) -> DebateResolution:
        """Call Gemini to synthesize the final resolution."""
        history = "DEBATE TRANSCRIPT:\n"
        for arg in transcript:
            history += f"--- {arg.agent_id} ---\nClaim: {arg.claim}\nLogic: {arg.reasoning}\nEvidence: {arg.evidence_or_logic}\n\n"
            
        content = f"TOPIC: {topic}\n\n{history}\nEvaluate and synthesize."
        
        return await self.llm.complete_json(
            schema=DebateResolution,
            system_prompt=JUDGE_PROMPT,
            user_input=content,
            temperature=0.2,
            request_kind="debate_judge",
        )
        
    async def _save_debate(
        self, topic: str, transcript: list[DebateArgument], resolution: DebateResolution
    ) -> None:
        """Persist the debate outcome."""
        debate_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        
        transcript_json = json.dumps([t.model_dump() for t in transcript])
        resolution_json = json.dumps(resolution.model_dump())
        
        await self.db.execute(
            """
            INSERT INTO debates (id, topic, transcript_json, resolution_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (debate_id, topic, transcript_json, resolution_json, created_at)
        )
        log.info(f"Debate {debate_id} saved.")

    async def track_uncertainty(self, topic: str, why_uncertain: str) -> None:
        """Log a topic that ARIA is genuinely uncertain about."""
        u_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        
        await self.db.execute(
            """
            INSERT INTO uncertainties (id, topic, why_uncertain, status, created_at)
            VALUES (?, ?, ?, 'open', ?)
            """,
            (u_id, topic, why_uncertain, created_at)
        )
        log.info(f"Uncertainty tracked: {topic}")

    async def get_uncertainties(self) -> list[UncertaintyTopic]:
        """Fetch all tracked uncertainties."""
        rows = await self.db.fetch_all("SELECT * FROM uncertainties WHERE status='open' ORDER BY created_at DESC")
        return [UncertaintyTopic(**dict(r)) for r in rows]
