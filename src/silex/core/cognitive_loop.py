"""
Cognitive Loop — ARIA's main reasoning cycle.

This is the heartbeat. Every user interaction flows through here:
  input → context build → Gemini call → state update → output

Phase 2: Now processes causal observations (builds graph), contradictions,
and hypotheses from every cognitive turn.
"""

from __future__ import annotations

import json
import uuid
import os
import asyncio
import errno
from datetime import datetime, timezone
from typing import Callable, Any

from silex.utils.telemetry import tracer

from silex.core.benchmark import BenchmarkRunner
from silex.core.context_builder import ContextBuilder
from silex.core.critic import ResponseCritic
from silex.core.creativity import CreativityStack
from silex.core.debate import DebateEngine
from silex.core.generalization import GeneralizationEngine
from silex.core.improver import ImprovementLogger
from silex.core.meta_reasoning import MetaReasoningEngine
from silex.core.planner import Planner
from silex.core.skills import SkillLoader
from silex.llm.catalog import list_providers
from silex.llm.factory import build_provider
from silex.llm.router import ModelRouter
from silex.memory.goal_tracker import GoalTracker
from silex.memory.memory_store import MemoryStore
from silex.memory.vector_store import VectorStore
from silex.memory.pruner import ContextPruner
from silex.memory.session import SessionManager
from silex.core.semantic_parser import SemanticParser
from silex.knowledge_graph.ontology import Ontology
from silex.models.schemas import (
    CausalEdge,
    CausalObservation,
    CognitiveResponse,
    Contradiction,
    HypothesisResolution,
    UncertaintyTrackingEntry,
    EdgeType,
    GoalUpdate,
    Hypothesis,
    KnowledgeNode,
    Memory,
    MemorySource,
    MemoryType,
    NewMemory,
    NodeType,
    Session,
    StoredContradiction,
    StoredHypothesis,
    VerificationStatus,
)
from silex.storage.database import Database
from silex.tools.registry import ToolRegistry
from silex.runtime.settings import RuntimeSettingsStore
from silex.runtime.usage import UsageTracker
from silex.utils.config import VYN_PROCESS_LOCK, VYN_ONTOLOGY, VYN_EXPORTS, VYN_HOME, WORKSPACE_DIR, autonomy_policy_snapshot
from silex.utils.config import allow_multi_writer, get_process_role, get_provider_settings, get_settings_store
from silex.utils.config import max_tool_calls_per_turn
from silex.utils.config import telegram_public_mode_enabled
from silex.utils.logger import setup_logger
from silex.utils.sanitize import sanitize_for_injection
from silex.world.contradictions import ContradictionDetector
from silex.world.graph import KnowledgeGraph
from silex.world.hypotheses import HypothesisEngine

log = setup_logger("silex.core")


class CognitiveLoop:
    """
    ARIA's main cognitive processing loop.

    Orchestrates: context building → LLM reasoning → state persistence.
    Phase 2 adds: graph building, contradiction detection, hypothesis tracking.
    """

    def __init__(self, assistant_name: str = "Silex"):
        self.assistant_name = assistant_name
        self.db = Database()
        self.settings_store: RuntimeSettingsStore = get_settings_store()
        self.usage_tracker = UsageTracker(self.db)
        self.memory = MemoryStore(self.db)
        self.goals = GoalTracker(self.db)
        self.session = SessionManager(self.db)
        self.planner = Planner(self.db)

        # Phase 19 — Memory Surface (MEMORY.md)
        from silex.memory.memory_surface import MemorySurface
        self.memory_surface = MemorySurface(self.memory)

        provider_settings = get_provider_settings(self.settings_store)
        self.gemini = build_provider(self.settings_store, self.usage_tracker)
        self.router = ModelRouter(
            fast_model=provider_settings["fast_model"],
            reasoning_model=provider_settings["reasoning_model"],
        )
        self._process_lock_path = VYN_PROCESS_LOCK

        # Phase 2 — World Model
        self.kg = KnowledgeGraph(self.db)
        self.contradictions = ContradictionDetector(self.db, self.kg)
        self.hypotheses = HypothesisEngine(self.db, self.kg)

        # Phase B: Milestone 2 — Vector Memory
        self.vector_store = VectorStore()
        self.pruner = ContextPruner(self.gemini)

        # Phase 5 — Tool Use
        self.tool_registry = ToolRegistry(
            vector_store=self.vector_store,
            db=self.db,
            session_manager=self.session,
            memory_store=self.memory,
            llm=self.gemini,
        )

        # Phase 6 — Generalization
        self.generalization_engine = GeneralizationEngine(self.gemini, self.db)

        # Phase C — Markdown Skills Ecosystem
        self.skill_loader = SkillLoader(vector_store=self.vector_store)
        self.skill_loader.load_all()
        self.creativity_stack = CreativityStack()

        # Phase 21 — Progressive Skill Disclosure: on-demand skill reading
        from silex.tools.skill_tool import ReadSkillTool
        self.tool_registry.register(ReadSkillTool(self.skill_loader))

        # Phase 21 — Background Review Daemon: post-turn learning
        from silex.core.background_review import BackgroundReviewer
        self.bg_reviewer = BackgroundReviewer(
            llm=self.gemini,
            skill_loader=self.skill_loader,
            memory_store=self.memory,
            db=self.db,
        )

        # Phase 7: Semantic Disambiguation
        self.ontology = Ontology()
        _ontology_overlay = VYN_ONTOLOGY
        if _ontology_overlay.is_file():
            try:
                self.ontology.merge_from_json_file(_ontology_overlay)
                log.info("Loaded ontology overlay from %s", _ontology_overlay)
            except Exception as exc:
                log.warning("Ontology overlay at %s was not loaded: %s", _ontology_overlay, exc)
        self.semantic_parser = SemanticParser(self.ontology)

        self.context_builder = ContextBuilder(
            self.memory, self.goals, self.session,
            knowledge_graph=self.kg,
            contradiction_detector=self.contradictions,
            hypothesis_engine=self.hypotheses,
            tool_registry=self.tool_registry,
            generalization_engine=self.generalization_engine,
            skill_loader=self.skill_loader,
            settings_store=self.settings_store,
            semantic_parser=self.semantic_parser, # Pass parser to context builder
            pruner=self.pruner,
            creativity_stack=self.creativity_stack,
            planner=self.planner,
            vector_store=self.vector_store,
        )

        # Phase 3 — Self-Improvement
        self.critic = ResponseCritic(self.gemini, model_override=provider_settings.get("critic_model"))
        self.improver = ImprovementLogger(self.db)

        # Phase 4 — Multi-Agent Debate
        self.debate_engine = DebateEngine(self.gemini, self.db)

        # Phase 7 — Recursive Self-Improvement
        self.meta_reasoning = MetaReasoningEngine(self.gemini, self.db)
        self.benchmark = BenchmarkRunner(self.gemini, self.db)

        # Wire meta_reasoning into context_builder for active directive injection
        self.context_builder.meta_reasoning = self.meta_reasoning
        # Wire the LLM provider for context window compression (C3)
        self.context_builder._llm_client = self.gemini

        # Wire context_builder into the SpawnWorkerTool
        if "spawn_worker" in self.tool_registry.tools:
            self.tool_registry.tools["spawn_worker"].context_builder = self.context_builder

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self, target_query: str | None = None) -> None:
        """Initialize all systems."""
        log.info("ARIA cognitive systems initializing...")
        self._acquire_process_lock()
        await self.db.connect()
        self.gemini.connect()

        # Phase 2: Load knowledge graph subgraph into memory
        await self.kg.load_relevant(target_query, max_nodes=200)

        # Phase B: Milestone 2 — Start background indexing (only when Chroma is available)
        if self.vector_store.is_active:
            from silex.memory.indexer import WorkspaceIndexer

            indexer = WorkspaceIndexer(self.vector_store, str(WORKSPACE_DIR))
            asyncio.create_task(asyncio.to_thread(indexer.run))

        # Phase 7: Load semantic profiles
        profiles = await self.memory.get_all_semantic_profiles()
        if profiles:
            self.semantic_parser.subjective_terms.update(profiles)
            log.info(f"Loaded {len(profiles)} custom semantic profiles.")

        await self.session.resume_or_start()
        # Step 0.25: Recovery Checkpoints on startup
        try:
            await self.recover_checkpoints()
        except Exception as e:
            log.warning(f"Failed to run startup recovery checkpoints: {e}")

        # Phase 21: Skill Curator — auto-run if ≥7 days since last curation
        try:
            from silex.core.curator import SkillCurator
            from silex.utils.config import VYN_SKILLS_ARCHIVE, VYN_CURATOR_LAST_RUN
            if SkillCurator.should_run(VYN_CURATOR_LAST_RUN):
                log.info("Skill Curator: ≥7 days since last run — starting background curation")
                curator = SkillCurator(self.skill_loader, VYN_SKILLS_ARCHIVE, llm=self.gemini)
                async def _run_curator():
                    await curator.run()
                    SkillCurator.record_run(VYN_CURATOR_LAST_RUN)
                asyncio.create_task(_run_curator())
        except Exception as e:
            log.warning(f"Skill Curator startup check failed (non-fatal): {e}")

        log.info("All systems online. Cognitive loop ready.")

    async def shutdown(self) -> None:
        """Gracefully shut down all systems."""
        log.info("ARIA shutting down...")
        await self.session.end_session()
        await self.db.close()
        self._release_process_lock()
        log.info("Shutdown complete.")

    async def recover_checkpoints(self) -> list[dict]:
        """
        Scan for any turn checkpoints left in 'executing_tools' status,
        log warning alerts, and return them for potential recovery.
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM turn_checkpoints WHERE status = 'executing_tools'"
        )
        recovered = []
        for r in rows:
            log.warning(
                f"🚨 MID-TURN CRASH DETECTED: Session {r['session_id']} turn {r['turn_number']} "
                f"was interrupted during tool execution! Draft reasoning: {r['draft_reasoning'][:100]}..."
            )
            recovered.append(dict(r))
        return recovered

    async def tick(self) -> None:
        """
        Background execution cycle. Called periodically by the server.
        Allows ARIA to act proactively without human prompting.
        """
        active_goals = await self.goals.get_active()
        if not active_goals:
            return
            
        target_goal = active_goals[0]
        
        system_prompt = await self.context_builder.build("BACKGROUND TICK")
        user_input = (
            f"[SYSTEM BACKGROUND EVENT] You have been woken up to work on your active goals in the background.\n"
            f"Your highest priority active goal is: '{target_goal.description}'.\n"
            f"Review this goal. Do you need to execute any tools (like search, file reading, or terminal commands) "
            f"to progress towards this goal? If yes, use your tools. "
            f"If no action is currently needed, simply state 'No action needed'."
        )
        
        try:
            cognitive = await self.gemini.think(system_prompt, user_input)
            if cognitive.tool_calls:
                log.info(f"⚡ PROACTIVE ACTION: ARIA executed {len(cognitive.tool_calls)} tools in the background.")
                results, any_failures, tool_results = await self._execute_tools(
                    cognitive.tool_calls,
                    None,
                    execution_mode="background",
                )
                
                # Re-draft to process the results and update memory
                tool_prompt = system_prompt + (
                    "\n\n═══════════════════════════════════════════════════════════\n"
                    "BACKGROUND TOOL RESULTS\n"
                    "═══════════════════════════════════════════════════════════\n"
                    "You executed tools in the background. Here are the results:\n\n"
                    f"{results}\n\n"
                    "Process these results, update your goals/graph if necessary, and log your thoughts."
                )
                await self.gemini.think(tool_prompt, "Process the background tool results.")
                
        except Exception as e:
            log.error(f"Error during background tick: {e}")

    # ------------------------------------------------------------------
    # The Loop
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        status_callback: Callable[..., Any] | None = None,
        images: list[dict] | None = None,
        response_callback: Callable[[str], None] | None = None,
    ) -> CognitiveResponse:
        """
        Process a single cognitive turn.

        Phase 5 flow:
          1. Build context
          2. Draft response / Plan tools (Gemini Pass 1)
          3. Execute tools if planned
          4. Re-draft with tool results (Gemini Pass 2)
          5. Critique draft
          6. If rejected, Retry (Gemini Pass 3)
          7. State updates
        """
        # Step -2: MEMORY.md startup sync (Phase 19 — runs once per session)
        if not getattr(self, '_memory_md_synced', False):
            self._memory_md_synced = True
            try:
                changes = await self.memory_surface.detect_user_edits()
                if changes:
                    log.info(f"Imported {len(changes)} memory edits from MEMORY.md")
            except Exception as e:
                log.warning(f"MEMORY.md sync failed (non-fatal): {e}")

        # Step -1: Check Semantic Response Cache
        try:
            cached_text = await self.session.get_cached_response(user_input)
            if cached_text is not None:
                log.info("Semantic Cache Interception: Query served from cache.")
                if status_callback:
                    status_callback("[dim]  (Engine: CACHE HIT)[/]")
                if response_callback:
                    response_callback("[START]")
                    response_callback(cached_text)
                return CognitiveResponse(
                    reasoning="Served from Semantic Cache",
                    response=cached_text,
                    new_memories=[],
                    goal_updates=[],
                    self_reflection="Cached query hit.",
                    confidence=1.0,
                    working_scratchpad=None,
                    uncertainty_flags=[],
                    uncertainty_tracking=[],
                    causal_observations=[],
                    contradictions_detected=[],
                    hypotheses=[],
                    hypothesis_resolutions=[],
                    tool_calls=[],
                    agent_delegation=[]
                )
        except Exception as e:
            log.warning(f"Failed to check response cache (non-fatal): {e}")
        async def _think(
            system_prompt: str,
            user_input: str,
            images: list[dict] | None = None,
            model_override: str | None = None,
        ) -> CognitiveResponse:
            if response_callback:
                response_callback("[START]")
                return await self.gemini.think_stream(
                    system_prompt,
                    user_input,
                    response_callback,
                    images=images,
                    model_override=model_override,
                )
            return await self.gemini.think(
                system_prompt,
                user_input,
                images=images,
                model_override=model_override,
            )
        # Step 0: Fast-Model Intent Routing
        try:
            provider_settings = get_provider_settings(self.settings_store)
            fast_model = provider_settings["fast_model"]
            
            router_prompt = (
                "You are VYN's Fast Intent Router.\n"
                "Evaluate the user's message. Does this user message require executing tools (like reading/writing files, run terminal commands, web search, browser), writing code, or deep logical/technical reasoning? Or is it simple conversational chitchat or trivial greetings (e.g. 'thanks', 'cool', 'hi', 'how are you')?\n"
                "Reply with exactly 'REASON' or 'CHAT'."
            )
            
            intent_response = await self.gemini.think(
                system_prompt=router_prompt,
                user_input=user_input,
                model_override=fast_model
            )
            
            intent = intent_response.response.strip().upper()
            log.info(f"Intent Routing: user input evaluated as {intent}")
            
            if "CHAT" in intent and "REASON" not in intent:
                # Fast conversational path
                if status_callback:
                    status_callback("[dim]  (Engine: FAST CHAT)[/]")
                    
                chat_prompt = (
                    "You are VYN, a highly capable cognitive AI assistant.\n"
                    "Provide a brief, helpful, and friendly conversational response to the user. "
                    "You do not have tools or full context active right now, so keep it strictly conversational. "
                    "Be fully in character. Make it brief."
                )
                
                chat_response = await _think(
                    system_prompt=chat_prompt,
                    user_input=user_input,
                    model_override=fast_model
                )
                
                # Persist turn to history
                await self.session.record_turn(
                    user_input=user_input,
                    reasoning="Conversational chitchat handled by Fast Model",
                    response=chat_response.response,
                    self_reflection="",
                    confidence=1.0,
                    memories_added=0,
                    goals_changed=0,
                    scratchpad=""
                )
                
                # Create a minimal CognitiveResponse
                return CognitiveResponse(
                    reasoning="Conversational chitchat handled by Fast Model",
                    response=chat_response.response,
                    self_reflection="",
                    confidence=1.0,
                    tool_calls=[],
                    new_memories=[],
                    goal_updates=[],
                    causal_observations=[],
                    contradictions_detected=[],
                    hypotheses=[],
                    hypothesis_resolutions=[],
                    uncertainty_tracking=[],
                    working_scratchpad=""
                )
        except Exception as e:
            log.warning(f"Fast intent routing failed: {e}. Falling back to normal reasoning flow.")

        # Step 0.5: Semantic Analysis
        semantic_analysis = self.semantic_parser.analyze_input(user_input)
        if semantic_analysis['subjective_interpretations']:
            log.info(f"Identified subjective terms: {list(semantic_analysis['subjective_interpretations'].keys())}")

        # Step 1: Build context (passing semantic analysis results)
        with tracer.start_as_current_span("build_context"):
            system_prompt = await self.context_builder.build(user_input, semantic_analysis=semantic_analysis)

        # Step 1.1: Skill usage tracking moved to ReadSkillTool.execute() (Phase 21)
        # Usage is now recorded only when the agent explicitly reads a skill,
        # not when skills are blindly matched and injected into the prompt.
        # Refinement checks are handled by the BackgroundReviewer (Step 14.5).

        try:
            # Step 1.5: Route (Determine Depth)
            target_model = self.router.route(user_input, context_size=len(system_prompt))
            if status_callback:
                provider_settings = get_provider_settings(self.settings_store)
                model_name = "REASONING" if target_model == provider_settings["reasoning_model"] else "FAST"
                status_callback(f"[dim]  (Engine: {model_name})[/]")

            # Step 2: Think (Pass 1)
            with tracer.start_as_current_span("llm_pass_1"):
                cognitive = await _think(system_prompt, user_input, images=images, model_override=target_model)
            
            # Step 2.1: Mid-Turn Crash Recovery Checkpointing
            if self.session.current:
                from datetime import datetime, timezone
                try:
                    await self.db.execute(
                        """
                        INSERT INTO turn_checkpoints (
                            session_id, turn_number, draft_reasoning, draft_plan, status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, turn_number) DO UPDATE SET
                            draft_reasoning=excluded.draft_reasoning,
                            status=excluded.status,
                            updated_at=excluded.updated_at
                        """,
                        (
                            self.session.current.id,
                            self.session.current.turn_count + 1,
                            cognitive.reasoning,
                            "", # draft_plan
                            "executing_tools",
                            datetime.now(timezone.utc).isoformat()
                        )
                    )
                except Exception as e:
                    log.warning(f"Checkpointing failed (non-fatal): {e}")

            # Step 2.5: Reasoning Consistency Check (Detect "Intent Drift")
            await self._check_reasoning_consistency(cognitive, status_callback=status_callback)

            plan_id = None
            if self.planner.should_plan(user_input, tool_count=len(cognitive.tool_calls)):
                plan = await self.planner.create_plan(
                    user_input=user_input,
                    session_id=self.session.current.id if self.session.current else None,
                    tool_names=[tc.tool_name for tc in cognitive.tool_calls],
                )
                plan_id = plan.id
            
            # Step 3: Tool Execution / Agent Delegation
            with tracer.start_as_current_span("tool_execution"):
                used_tools = bool(cognitive.tool_calls)
                has_delegation = bool(getattr(cognitive, "agent_delegation", None))
                tool_prompt = system_prompt  # default; overwritten if tools or delegation are used
                tool_failures = False
                
                tool_results_text = ""
                delegation_results_text = ""
                
                if used_tools:
                    if status_callback:
                        status_callback(f"[magenta]  Executing {len(cognitive.tool_calls)} tools...[/]")
                    
                    tool_results_text, any_failures, tool_results = await self._execute_tools(
                        cognitive.tool_calls,
                        status_callback,
                        execution_mode="interactive",
                    )
                    tool_failures = any_failures
                    await self.planner.reconcile_tools(plan_id, tool_results)
                    
                    # --- Milestone 4: Self-Healing Loop (Immune System) ---
                    max_healing_attempts = 2
                    attempt = 0
                    while any_failures and attempt < max_healing_attempts:
                        attempt += 1
                        if status_callback:
                            status_callback(f"[yellow]  ⚠ Tool failure detected. Triggering Self-Healing (Attempt {attempt})...[/]")
                        
                        safe_healing_results = sanitize_for_injection(tool_results_text)
                        healing_prompt = system_prompt + (
                            "\n\n═══════════════════════════════════════════════════════════\n"
                            "IMMUNE SYSTEM: SELF-HEALING PROTOCOL\n"
                            "═══════════════════════════════════════════════════════════\n"
                            "The following tool calls failed with errors. "
                            "You MUST analyze the errors, fix the cause (e.g., via code_editor or run_terminal_command), "
                            "and retry the necessary actions.\n\n"
                            f"{safe_healing_results}\n\n"
                            "Your mission is to resolve these failures autonomously. DO NOT ask the user for help."
                        )
                        
                        # Pass the healing prompt to Gemini to get correction tool calls
                        healing_cognitive = await self.gemini.think(healing_prompt, user_input, model_override=target_model)
                        
                        if not healing_cognitive.tool_calls:
                            log.warning("Self-healing triggered but model provided no further tools.")
                            break
                            
                        # Execute the healing tools
                        healing_results, any_failures, healing_tool_results = await self._execute_tools(
                            healing_cognitive.tool_calls,
                            status_callback,
                            execution_mode="interactive",
                        )
                        tool_failures = tool_failures or any_failures
                        await self.planner.reconcile_tools(plan_id, healing_tool_results)
                        # Accumulate results
                        tool_results_text += "\n" + healing_results

                if has_delegation:
                    if status_callback:
                        status_callback(f"[magenta]  Spawning {len(cognitive.agent_delegation)} parallel worker agents...[/]")
                    
                    from silex.core.agent import AgentOrchestrator
                    orchestrator = AgentOrchestrator(
                        llm=self.gemini,
                        tool_registry=self.tool_registry,
                        context_builder=self.context_builder,
                        db=self.db
                    )
                    
                    # Execute worker agents concurrently
                    agent_results = await orchestrator.execute_agents(cognitive.agent_delegation)
                    
                    # Merge their results via Lead Coordinator pass
                    merged_output = await orchestrator.merge_results(agent_results, user_input)
                    
                    # Format synthesized results
                    delegation_results_text = (
                        "═══════════════════════════════════════════════════════════\n"
                        "PARALLEL WORKER DELEGATION RESULTS\n"
                        "═══════════════════════════════════════════════════════════\n"
                        f"Cohesive Response from Workers:\n{merged_output.response}\n\n"
                    )
                    if merged_output.dissent_summary:
                        delegation_results_text += f"Dissent / Caveats Summary:\n{merged_output.dissent_summary}\n\n"
                    if merged_output.proposed_edits:
                        delegation_results_text += f"Proposed Edits:\n{json.dumps(merged_output.proposed_edits, indent=2)}\n\n"
                    if merged_output.new_observations:
                        delegation_results_text += f"Causal Observations:\n{json.dumps(merged_output.new_observations, indent=2)}\n\n"
                
                # Step 4: Re-draft with tool and/or delegation results
                if used_tools or has_delegation:
                    if status_callback:
                        status_callback("[bright_cyan]  Observing results and re-drafting...[/]")
                        
                    prompt_additions = ""
                    if used_tools:
                        safe_tool_results = sanitize_for_injection(tool_results_text)
                        prompt_additions += (
                            "═══════════════════════════════════════════════════════════\n"
                            "TOOL EXECUTION RESULTS\n"
                            "═══════════════════════════════════════════════════════════\n"
                            "You requested to use tools. Here are the cumulative results:\n\n"
                            f"{safe_tool_results}\n\n"
                        )
                    if has_delegation:
                        safe_delegation_results = sanitize_for_injection(delegation_results_text)
                        prompt_additions += safe_delegation_results
                        prompt_additions += (
                            "IMPORTANT: You MUST incorporate any approved worker delegation findings, "
                            "proposed edits, and causal observations into your final response. "
                            "Populate the `proposed_edits` and `causal_observations` in your own JSON "
                            "response if they are required to implement the task.\n\n"
                        )
                        
                    tool_prompt = system_prompt + (
                        f"\n\n{prompt_additions}"
                        "Now, incorporate these facts into your final response."
                    )
                    cognitive = await _think(tool_prompt, user_input, model_override=target_model)

            # Step 5: Critique
            with tracer.start_as_current_span("critic_evaluation"):
                if status_callback:
                    status_callback("[bright_cyan]  Critiquing draft...[/]")
                    
                # The context should include tool results if they were run
                current_context = tool_prompt if (used_tools or has_delegation) else system_prompt
                
                critique = await self.critic.critique(
                    user_input=user_input,
                    system_context=current_context,
                    draft_response=cognitive.response,
                    draft_reasoning=cognitive.reasoning,
                )
                
                # Step 6: Critique → closed retry loop (max 3 attempts, keep best)
                MAX_CRITIC_ATTEMPTS = 3
                best_cognitive = cognitive
                best_score = sum([
                    critique.scores.accuracy,
                    critique.scores.depth,
                    critique.scores.honesty,
                ])
                attempt_num = 1
    
                while not critique.is_acceptable and attempt_num < MAX_CRITIC_ATTEMPTS:
                    attempt_num += 1
                    if status_callback:
                        status_callback(
                            f"[yellow]  ⚠ Draft rejected (Acc:{critique.scores.accuracy:.1f}, "
                            f"Dep:{critique.scores.depth:.1f}, "
                            f"Hon:{critique.scores.honesty:.1f}). "
                            f"Retrying (attempt {attempt_num}/{MAX_CRITIC_ATTEMPTS})...[/]"
                        )
    
                    retry_prompt = current_context + (
                        "\n\n═══════════════════════════════════════════════════════════\n"
                        "CRITIQUE OF PREVIOUS DRAFT\n"
                        "═══════════════════════════════════════════════════════════\n"
                        "Your previous draft was rejected by the Internal Critic for the following reasons:\n"
                        f"{critique.feedback}\n\n"
                        "Do NOT apologize. Do NOT mention the critic. Just output a better response "
                        "that fixes these specific issues."
                    )
    
                    retry_cognitive = await _think(retry_prompt, user_input, model_override=target_model)
    
                    # Step 6.25: Fix Critic Retry Loop "Tool Ignorance"
                    retry_context = retry_prompt
                    if retry_cognitive.tool_calls:
                        if status_callback:
                            status_callback(f"[magenta]  Critic Retry: Executing {len(retry_cognitive.tool_calls)} planned tools...[/]")
                        
                        retry_tool_results_text, any_failures, retry_tool_results = await self._execute_tools(
                            retry_cognitive.tool_calls,
                            status_callback,
                            execution_mode="interactive",
                        )
                        await self.planner.reconcile_tools(plan_id, retry_tool_results)
                        
                        safe_retry_tool_results = sanitize_for_injection(retry_tool_results_text)
                        retry_context = retry_prompt + (
                            "\n\n═══════════════════════════════════════════════════════════\n"
                            "CRITIC RETRY: TOOL EXECUTION RESULTS\n"
                            "═══════════════════════════════════════════════════════════\n"
                            "You requested to use tools during this critique retry. Here are the results:\n\n"
                            f"{safe_retry_tool_results}\n\n"
                            "Incorporate these facts into your updated final response."
                        )
                        retry_cognitive = await _think(retry_context, user_input, model_override=target_model)
    
                    # Re-run critic on the new attempt
                    retry_critique = await self.critic.critique(
                        user_input=user_input,
                        system_context=retry_context,
                        draft_response=retry_cognitive.response,
                        draft_reasoning=retry_cognitive.reasoning,
                    )
    
                    retry_score = sum([
                        retry_critique.scores.accuracy,
                        retry_critique.scores.depth,
                        retry_critique.scores.honesty,
                    ])
    
                    # Log improvement attempt
                    if self.session.current:
                        await self._log_failure("critic_rejection", f"Attempt {attempt_num} rejected: {critique.feedback[:100]}")
                        await self.improver.log_improvement(
                            session_id=self.session.current.id,
                            turn_number=self.session.current.turn_count + 1,
                            draft=cognitive,
                            critique=critique,
                            final=retry_cognitive,
                        )
    
                    # Keep the best-scoring attempt (not just the last)
                    if retry_score > best_score:
                        best_cognitive = retry_cognitive
                        best_score = retry_score
    
                    cognitive = retry_cognitive
                    critique = retry_critique
    
                    if status_callback:
                        status_callback(f"[bright_cyan]  Attempt {attempt_num} complete (score: {retry_score:.2f}).[/]")
    
                # Accept best attempt found across all retries
                if best_cognitive is not cognitive:
                    log.info(f"Critic: accepting best attempt (score {best_score:.2f}) over final attempt.")
                cognitive = best_cognitive

            await self.planner.complete_plan(plan_id, blocked=tool_failures)

        except json.JSONDecodeError as e:
            log.error(f"JSON parsing failed: {e}")
            cognitive = self._make_error_response(
                "I received a malformed response from my reasoning engine. Retrying on next turn."
            )
        except ValueError as e:
            log.error(f"Value error in cognitive loop: {e}")
            cognitive = self._make_error_response(
                "I encountered a data validation error. Please try rephrasing your input."
            )
        except Exception as e:
            log.error(f"Cognitive loop failed: {e}", exc_info=True)
            # SECURITY: Do NOT include raw exception in user-facing response
            cognitive = self._make_error_response(
                "I'm having trouble processing that right now. Your input was received "
                "and I'll try again on the next turn."
            )

        # Batch all persistence operations into a single atomic transaction
        async with self.db.transaction():
            # Step 7: Persist new memories
            memories_added = await self._store_memories(cognitive.new_memories)

            # Step 8: Process goal updates
            goals_changed = await self._process_goals(cognitive.goal_updates)

            # Step 9: Build knowledge graph from causal observations
            graph_updates = await self._process_causal_observations(
                cognitive.causal_observations
            )

            # Step 9.5: Abstract principles from new observations (Phase 6)
            if graph_updates > 0 and cognitive.causal_observations:
                try:
                    await self.generalization_engine.abstract_principles(
                        cognitive.causal_observations
                    )
                except Exception as e:
                    log.warning(f"Principle extraction failed (non-fatal): {e}")

            # Step 10: Process contradictions
            await self._process_contradictions(
                cognitive.contradictions_detected
            )

            # Step 11: Store hypotheses
            await self._process_hypotheses(cognitive.hypotheses)

            # Step 11.25: Resolve hypotheses when the model (or operator path) supplies resolutions
            await self._process_hypothesis_resolutions(cognitive.hypothesis_resolutions)

            # Step 11.4: Record explicit uncertainty topics (Phase 4 — uncertainties table)
            await self._process_uncertainty_tracking(cognitive.uncertainty_tracking)

            # Step 11.5: Process self-improvement proposals (Phase 7 — Safety Locked)
            if getattr(cognitive, "inline_proposals", None) and self.meta_reasoning and self.session.current:
                try:
                    await self.meta_reasoning.process_inline_proposals(
                        cognitive.inline_proposals,
                        self.session.current.id,
                    )
                except Exception as e:
                    log.warning(f"Proposal processing failed (non-fatal): {e}")

            # Step 12: Record this turn
            await self.session.record_turn(
                user_input=user_input,
                reasoning=cognitive.reasoning,
                response=cognitive.response,
                self_reflection=cognitive.self_reflection,
                confidence=cognitive.confidence,
                memories_added=memories_added,
                goals_changed=goals_changed,
                scratchpad=getattr(cognitive, "working_scratchpad", None),
            )

            # Step 12.5: Cleanup turn checkpoint
            if self.session.current:
                await self.db.execute(
                    "DELETE FROM turn_checkpoints WHERE session_id = ? AND turn_number = ?",
                    (self.session.current.id, self.session.current.turn_count)
                )

        # Step 13: Store in response cache if it was a clean response (no tool_calls or agent_delegation)
        if (
            cognitive.response
            and not getattr(cognitive, "tool_calls", None)
            and not getattr(cognitive, "agent_delegation", None)
        ):
            try:
                await self.session.store_cached_response(user_input, cognitive.response)
            except Exception as e:
                log.warning(f"Failed to store response in cache (non-fatal): {e}")

        # Step 14: Volume-Based Active Hibernation trigger
        if await self.memory.check_consolidation_trigger():
            log.info("Volume trigger hit. Spawning background memory consolidation task.")
            asyncio.create_task(self._run_background_consolidation())

        # Step 14.5: Background Review Daemon (Phase 21 — post-turn learning)
        if self.session.current and hasattr(self, "bg_reviewer"):
            try:
                recent = await self.session.get_recent_turns(limit=6)
                asyncio.create_task(
                    self.bg_reviewer.maybe_review(recent, self.session.current.id)
                )
            except Exception as e:
                log.warning(f"Background review dispatch failed (non-fatal): {e}")

        # Step 15: Session reflection (Phase 19 — every 10 turns)
        if self.session.current and self.session.current.turn_count % 10 == 0:
            asyncio.create_task(self._reflect_on_session())

        return cognitive

    # ------------------------------------------------------------------
    # Phase 19 — Session Reflection
    # ------------------------------------------------------------------

    async def _reflect_on_session(self) -> None:
        """End-of-session reflective analysis.

        Runs as a background task every 10 turns. Performs two actions:
        1. If ≥2 critic rejections occurred this session, triggers meta-reasoning
        2. Creates a Level 2 session summary memory with topic extraction
        """
        if not self.session.current or self.session.current.turn_count < 5:
            return

        try:
            session = self.session.current

            # 1. Check for enough critic rejections to warrant meta-analysis
            recent_improvements = await self.improver.get_recent_improvements(limit=10)
            session_improvements = [
                imp for imp in recent_improvements
                if imp.session_id == session.id
            ]

            if len(session_improvements) >= 2 and self.meta_reasoning:
                log.info("Session reflection: ≥2 critic rejections detected, running meta-analysis...")
                proposal = await self.meta_reasoning.analyze_and_propose()
                if proposal:
                    log.info(f"Meta-reasoning proposal generated: {proposal.description[:60]}...")

            # 2. Generate a session summary memory (Level 2 — synthesized)
            turns = await self.session.get_recent_turns(limit=session.turn_count)
            if turns:
                topics = set()
                for t in turns:
                    words = [w.lower() for w in t.user_input.split() if len(w) > 4]
                    topics.update(words[:3])

                summary_content = (
                    f"Session summary ({session.turn_count} turns): "
                    f"Topics discussed: {', '.join(list(topics)[:10])}. "
                    f"Critic rejections: {len(session_improvements)}. "
                    f"Avg confidence: {session.avg_confidence:.2f}."
                )

                summary_memory = Memory(
                    content=summary_content,
                    source=MemorySource.REFLECTION,
                    importance=0.4,
                    level=2,
                    tags=["session_summary"],
                )
                await self.memory.add(summary_memory)
                log.info("Session reflection: Summary memory stored")

            # 3. Regenerate MEMORY.md with latest state
            try:
                await self.memory_surface.generate()
            except Exception as e:
                log.warning(f"MEMORY.md generation failed (non-fatal): {e}")

        except Exception as e:
            log.warning(f"Session reflection failed (non-fatal): {e}")

    async def _check_reasoning_consistency(
        self, cognitive: CognitiveResponse, status_callback: Callable[..., Any] | None = None
    ) -> None:
        """Verify that tool_calls match the intent described in reasoning."""
        if not cognitive.tool_calls and "tool" not in cognitive.reasoning.lower():
            return

        tool_names = [tc.tool_name for tc in cognitive.tool_calls]
        # Check if reasoning mentions tools but none were called, or vice-versa
        mentioned_tool = any(word in cognitive.reasoning.lower() for word in ["call", "use", "run", "search", "browse"])
        
        has_mismatch = False
        if cognitive.tool_calls and not mentioned_tool:
            has_mismatch = True
            log.warning(f"Reasoning Consistency: Model called tools {tool_names} but reasoning does not mention tool use.")
        elif not cognitive.tool_calls and mentioned_tool and len(cognitive.reasoning) > 50:
            # Only flag if reasoning is substantial (prevents false positives on "I don't need tools")
            if any(word in cognitive.reasoning.lower() for word in ["will call", "decided to use", "need to search"]):
                has_mismatch = True
                log.warning("Reasoning Consistency: Model reasoning indicates tool use, but no tool_calls were generated.")

        if has_mismatch and self.session.current:
            # Log this as a "soft failure" for the meta-reasoning analyst
            await self.db.execute(
                """
                INSERT INTO uncertainties (id, topic, why_uncertain, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    "Reasoning Consistency",
                    f"Reasoning vs Tools mismatch. Reasoning: {cognitive.reasoning[:100]}... Tools: {tool_names}",
                    "open",
                    datetime.now(timezone.utc).isoformat()
                )
            )
            if status_callback:
                status_callback("[yellow]  ⚠ Reasoning consistency mismatch detected and logged.[/]")
            await self._log_failure("consistency_mismatch", f"Reasoning vs Tools mismatch: {tool_names}")

    async def _log_failure(self, failure_type: str, description: str) -> None:
        """Log a failure for recent context window awareness."""
        if not self.session.current:
            return
        await self.db.execute(
            """
            INSERT INTO recent_failures (id, session_id, failure_type, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                self.session.current.id,
                failure_type,
                description,
                datetime.now(timezone.utc).isoformat()
            )
        )

    @staticmethod
    def _make_error_response(user_message: str) -> CognitiveResponse:
        """Create a safe error CognitiveResponse without leaking internals."""
        return CognitiveResponse(
            reasoning="My reasoning engine encountered an internal error.",
            response=user_message,
            new_memories=[],
            goal_updates=[],
            self_reflection="Failed to reason. Need to investigate the error.",
            confidence=0.0,
            working_scratchpad=None,
            uncertainty_flags=["internal_error"],
            uncertainty_tracking=[],
            causal_observations=[],
            contradictions_detected=[],
            hypotheses=[],
            hypothesis_resolutions=[],
            tool_calls=[],
        )

    @staticmethod
    def _format_tool_result(call, result) -> str:
        """Format tool result for inclusion in the cognitive response prompt."""
        text = f"--- Tool: {call.tool_name} ---\n"
        text += f"Expected: {call.expected_outcome}\n"
        if result.ethical_decision:
            text += (
                "Ethical Decision: "
                f"{result.ethical_decision.action.value} "
                f"({result.ethical_decision.principle})\n"
            )
        safe_outcome = sanitize_for_injection(result.actual_outcome)
        text += f"Actual Result:\n<tool_output>\n{safe_outcome}\n</tool_output>\n\n"
        return text

    async def _log_tool_execution(self, call, result) -> None:
        """Log a tool execution to the database action_logs, and log failure if not successful."""
        args_dict = {}
        if isinstance(call.arguments, str):
            try:
                args_dict = json.loads(call.arguments)
            except json.JSONDecodeError:
                log.warning(f"Failed to parse tool arguments: {call.arguments}")
        elif isinstance(call.arguments, dict):
            args_dict = call.arguments
            
        ethical_summary = "No ethical review recorded"
        if result.ethical_decision:
            ethical_summary = (
                f"{result.ethical_decision.action.value} via "
                f"{result.ethical_decision.principle}"
            )

        if self.session.current:
            tool_def = self.tool_registry.tools.get(call.tool_name)
            risk_level = tool_def.risk_level if tool_def else "unknown"
            log_id = str(uuid.uuid4())
            await self.db.execute(
                """
                INSERT INTO action_logs (
                    id, session_id, turn_number, tool_name, arguments_json,
                    expected_outcome, actual_outcome, success, risk_level,
                    model_update, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    self.session.current.id,
                    self.session.current.turn_count + 1,
                    call.tool_name,
                    json.dumps(args_dict),
                    call.expected_outcome,
                    result.actual_outcome,
                    result.success,
                    risk_level,
                    f"Ethical decision: {ethical_summary}. Update pending",
                    datetime.now(timezone.utc).isoformat()
                )
            )
        
        if not result.success:
            await self._log_failure("tool_error", f"Tool {call.tool_name} failed: {result.actual_outcome[:100]}")

    async def _execute_tools(self, tool_calls, status_callback, execution_mode: str = "interactive"):
        """Execute tool calls and return formatted text, failure flag, and raw results."""
        results_text = ""
        any_failures = False
        tool_results = []
        budget = max_tool_calls_per_turn()
        if len(tool_calls) > budget:
            any_failures = True
            results_text += f"Error: Tool budget exceeded ({len(tool_calls)} requested, max {budget}).\n\n"
            tool_calls = tool_calls[:budget]

        # Partition tool calls into batches of consecutive read-only tools and single write/network tools
        batches = []
        current_parallel_batch = []
        for call in tool_calls:
            tool_def = self.tool_registry.tools.get(call.tool_name)
            is_read_only = tool_def is not None and tool_def.risk_level == "read_only"
            if is_read_only:
                current_parallel_batch.append(call)
            else:
                if current_parallel_batch:
                    batches.append({"type": "parallel", "calls": current_parallel_batch})
                    current_parallel_batch = []
                batches.append({"type": "sequential", "call": call})
        if current_parallel_batch:
            batches.append({"type": "parallel", "calls": current_parallel_batch})

        # Execute batches
        for batch in batches:
            if batch["type"] == "parallel":
                calls = batch["calls"]
                if len(calls) == 1:
                    # Single read-only tool, execute normally
                    call = calls[0]
                    if status_callback:
                        status_callback(f"[magenta]  Running: {call.tool_name}...[/]")
                    result = await self.tool_registry.execute(call, execution_mode=execution_mode)
                    tool_results.append(result)
                    await self._log_tool_execution(call, result)
                    results_text += self._format_tool_result(call, result)
                    if not result.success:
                        any_failures = True
                else:
                    # Concurrently execute consecutive read-only tools
                    if status_callback:
                        names = ", ".join(c.tool_name for c in calls)
                        status_callback(f"[magenta]  Running concurrently: {names}...[/]")
                    
                    tasks = [
                        self.tool_registry.execute(call, execution_mode=execution_mode)
                        for call in calls
                    ]
                    results = await asyncio.gather(*tasks)
                    
                    for call, result in zip(calls, results):
                        tool_results.append(result)
                        await self._log_tool_execution(call, result)
                        results_text += self._format_tool_result(call, result)
                        if not result.success:
                            any_failures = True
            else:
                # Write/sequential tool execution
                call = batch["call"]
                if status_callback:
                    status_callback(f"[magenta]  Running: {call.tool_name}...[/]")
                
                result = await self.tool_registry.execute(call, execution_mode=execution_mode)
                tool_results.append(result)
                await self._log_tool_execution(call, result)
                results_text += self._format_tool_result(call, result)
                if not result.success:
                    any_failures = True
                
        return results_text, any_failures, tool_results


    async def _run_background_consolidation(self) -> None:
        """Runs memory consolidation in a background task with error containment."""
        try:
            await self.pruner.consolidate_memories(self.memory)
        except Exception as e:
            log.error(f"Background memory consolidation failed: {e}", exc_info=True)

    async def _store_memories(self, new_memories: list[NewMemory]) -> int:
        """Persist new memories from the cognitive response."""
        count = 0
        for nm in new_memories:
            try:
                source = MemorySource(nm.source.strip().lower())
            except (ValueError, KeyError, AttributeError):
                source = MemorySource.INFERENCE
            try:
                memory_type = MemoryType(nm.memory_type.strip().lower())
            except (ValueError, KeyError, AttributeError):
                memory_type = MemoryType.SEMANTIC

            memory = Memory(
                content=nm.content,
                source=source,
                memory_type=memory_type,
                importance=nm.importance,
                confidence=nm.confidence,
                tags=nm.tags,
                provenance={
                    "session_id": self.session.current.id if self.session.current else None,
                    "turn_number": (self.session.current.turn_count + 1) if self.session.current else None,
                    "memory_type": memory_type.value,
                    "identity_relevant": memory_type in {MemoryType.NORMATIVE, MemoryType.CHARACTER},
                    "requires_review": memory_type == MemoryType.NORMATIVE,
                    "source_kind": source.value,
                },
            )
            saved_mem = await self.memory.add(memory)
            if saved_mem is not None:
                count += 1

        if count > 0:
            log.debug(f"Stored {count} new memories")
        return count

    async def _process_goals(self, goal_updates: list[GoalUpdate]) -> int:
        """Process goal updates from the cognitive response.

        Safety: A cooldown of GOAL_COOLDOWN_SECONDS is enforced between
        goal state transitions (create/complete/abandon) to prevent the
        background loop from burning API tokens in a create-complete cycle.
        """
        now = datetime.now(timezone.utc)
        cooldown_seconds = int(os.environ.get("ARIA_GOAL_COOLDOWN_SECONDS", "600"))

        # Check cooldown — skip state transitions if the last one was too recent
        if hasattr(self, "_last_goal_transition") and self._last_goal_transition:
            elapsed = (now - self._last_goal_transition).total_seconds()
            if elapsed < cooldown_seconds:
                log.debug(
                    f"Goal cooldown active ({int(cooldown_seconds - elapsed)}s remaining). "
                    f"Skipping {len(goal_updates)} goal updates."
                )
                return 0

        count = 0
        for update in goal_updates:
            try:
                if update.action == "create":
                    await self.goals.create(
                        description=update.description,
                        priority=update.priority,
                    )
                    count += 1
                    log.info(f"🎯 Goal created: {update.description}")
                elif update.action == "complete":
                    goal = await self.goals.find_by_description(update.description)
                    if goal:
                        await self.goals.complete(goal.id, notes=update.notes)
                        count += 1
                        log.info(f"✅ Goal completed: {update.description}")
                        
                        # Trigger auto-skill synthesis in the background
                        if len(update.description) > 10:
                            asyncio.create_task(self._synthesize_skill(update.description))
                elif update.action == "abandon":
                    goal = await self.goals.find_by_description(update.description)
                    if goal:
                        await self.goals.abandon(goal.id, notes=update.notes)
                        count += 1
                        log.info(f"🚫 Goal abandoned: {update.description}")
                elif update.action == "update":
                    log.debug(f"Goal update noted: {update.description}")
            except Exception as e:
                log.warning(f"Failed to process goal update: {e}")

        if count > 0:
            self._last_goal_transition = now
            log.info(f"Processed {count} goal state transitions")
        return count

    async def _synthesize_skill(self, goal_description: str) -> None:
        """Background task: synthesize a reusable skill after goal completion.

        Phase 19 upgrades:
        - Quality gate: requires ≥3 session turns and non-trivial goal
        - Deduplication: checks existing skills for semantic similarity
        - Uses fast model (cheap) for initial drafts
        - Structured output via SkillSynthesisResponse schema
        - Generates proper YAML frontmatter
        - Progressive refinement: skills earn quality investment via reuse
        """
        try:
            from silex.core.skill_schemas import SkillSynthesisResponse

            # ── Quality gate ──────────────────────────────────────────
            if len(goal_description) <= 20:
                log.debug(f"Skill synthesis skipped: goal too short ({len(goal_description)} chars)")
                return

            if self.session.current and self.session.current.turn_count < 3:
                log.debug("Skill synthesis skipped: session has < 3 turns")
                return

            # ── Deduplication ─────────────────────────────────────────
            existing = self.skill_loader.find_similar(goal_description)
            if existing:
                log.info(f"Skill synthesis skipped: similar skill '{existing}' already exists")
                return

            # ── Gather session history (efficient join) ───────────────
            recent_turns = await self.session.get_recent_turns(limit=20)
            if not recent_turns:
                return

            history_text = "\n\n".join(
                f"USER: {t.user_input}\nARIA: {t.response}"
                for t in recent_turns
            )

            # ── Synthesize via fast model + structured output ─────────
            prompt = (
                "You just completed a goal for the user. Analyze the session history "
                "and synthesize the workflow into a reusable Skill document.\n\n"
                "Guidelines:\n"
                "- The skill_name should be a short, hyphenated slug (e.g., 'deploy-docker-container')\n"
                "- The description should start with 'Use when...' and be one line\n"
                "- The steps should be a generic, reusable Markdown tutorial\n"
                "- Strip out project-specific details — make it applicable to any similar task\n"
                "- Set is_generalizable to false if the workflow was too project-specific to reuse\n"
            )

            provider_settings = get_provider_settings(self.settings_store)
            synthesis = await self.gemini.complete_json(
                schema=SkillSynthesisResponse,
                system_prompt=prompt,
                user_input=f"Goal Completed: {goal_description}\n\nSession History:\n{history_text}",
                model_override=provider_settings.get("fast_model"),
                temperature=0.3,
                request_kind="skill_synthesis",
            )

            # ── Generalizability check ────────────────────────────────
            if not synthesis.is_generalizable:
                log.info(f"Skill synthesis skipped: LLM judged '{synthesis.skill_name}' as not generalizable")
                return

            # ── Build skill file with YAML frontmatter ────────────────
            from datetime import datetime, timezone as tz
            from silex.utils.config import VYN_SKILLS
            import re as _re

            now_iso = datetime.now(tz.utc).isoformat()
            slug = _re.sub(r"[^a-z0-9-]", "-", synthesis.skill_name.lower().strip())
            slug = _re.sub(r"-+", "-", slug).strip("-")[:60]
            if not slug:
                slug = "unnamed-skill"

            frontmatter = (
                f"---\n"
                f"name: {slug}\n"
                f"description: \"{synthesis.description}\"\n"
                f"version: 1\n"
                f"uses: 0\n"
                f"created_at: \"{now_iso}\"\n"
                f"last_used: \"\"\n"
                f"refined: false\n"
                f"---\n"
            )

            prereqs = ""
            if synthesis.prerequisites:
                prereqs = "\n## Prerequisites\n" + "\n".join(
                    f"- {p}" for p in synthesis.prerequisites
                ) + "\n"

            skill_content = f"{frontmatter}\n# {synthesis.skill_name}\n\n{synthesis.description}\n{prereqs}\n{synthesis.steps}"

            skill_path = VYN_SKILLS / f"{slug}.md"
            skill_path.write_text(skill_content, encoding="utf-8")
            log.info(f"✨ Auto-Skill Synthesized: {skill_path.name} for goal '{goal_description}'")

            # Reload skills so the new one is immediately available
            self.skill_loader.load_all()

        except Exception as e:
            log.error(f"Failed to synthesize auto-skill for '{goal_description}': {e}")

    async def _refine_skill(self, skill_name: str) -> None:
        """Refine a skill that has been used 2+ times using the reasoning model.

        This is the progressive quality investment: skills start cheap (fast model),
        and earn a reasoning-model polish pass once proven useful through reuse.
        """
        try:
            if skill_name not in self.skill_loader.skills:
                return

            content = self.skill_loader.skills[skill_name]
            _meta_dict, body = self.skill_loader.parse_frontmatter(content)

            provider_settings = get_provider_settings(self.settings_store)
            prompt = (
                "You are refining an AI agent skill document. The skill has been used "
                "multiple times and proven useful. Your job is to improve it:\n"
                "1. Make the steps more precise and complete\n"
                "2. Add edge cases and error handling guidance\n"
                "3. Improve clarity and remove ambiguity\n"
                "4. Make it more generalizable (strip remaining project-specific details)\n"
                "5. Keep the same structure but enhance quality\n\n"
                "Output ONLY the improved Markdown body (no frontmatter, no fences)."
            )

            response = await self.gemini.think(
                system_prompt=prompt,
                user_input=f"Current skill content:\n\n{body}",
                model_override=provider_settings.get("reasoning_model"),
            )

            refined_body = response.response.strip()
            # Strip markdown fences if present
            if refined_body.startswith("```"):
                refined_body = refined_body.split("\n", 1)[-1]
            if refined_body.endswith("```"):
                refined_body = refined_body[:-3].strip()

            success = self.skill_loader.update_skill(
                skill_name, refined_body, extra_meta={"refined": True}
            )
            if success:
                log.info(f"🔬 Skill refined: {skill_name}")

        except Exception as e:
            log.error(f"Failed to refine skill '{skill_name}': {e}")

    # ------------------------------------------------------------------
    # Phase 2 — World Model Processing
    # ------------------------------------------------------------------

    async def _process_causal_observations(
        self, observations: list[CausalObservation]
    ) -> int:
        """
        Process causal observations from Gemini into the knowledge graph.

        For each observation:
          1. Find or create the source node
          2. Find or create the target node
          3. Create a typed edge between them
        """
        count = 0
        for obs in observations:
            try:
                # Find or create source node
                src_id = self.kg.find_node_by_content(obs.from_concept)
                if not src_id:
                    src_node = KnowledgeNode(
                        content=obs.from_concept,
                        node_type=NodeType.CONCEPT,
                        confidence=obs.strength,
                        source="inference",
                        verification_status=VerificationStatus.UNVERIFIED,
                        metadata={"provenance": "cognitive_observation"},
                    )
                    src_node = await self.kg.add_node(src_node)
                    src_id = src_node.id

                # Find or create target node
                tgt_id = self.kg.find_node_by_content(obs.to_concept)
                if not tgt_id:
                    tgt_node = KnowledgeNode(
                        content=obs.to_concept,
                        node_type=NodeType.CONCEPT,
                        confidence=obs.strength,
                        source="inference",
                        verification_status=VerificationStatus.UNVERIFIED,
                        metadata={"provenance": "cognitive_observation"},
                    )
                    tgt_node = await self.kg.add_node(tgt_node)
                    tgt_id = tgt_node.id

                # Create the edge
                try:
                    edge_type = EdgeType(obs.relationship)
                except ValueError:
                    edge_type = EdgeType.SUPPORTS  # fallback

                edge = CausalEdge(
                    source_node=src_id,
                    target_node=tgt_id,
                    edge_type=edge_type,
                    strength=obs.strength,
                    evidence=obs.evidence,
                )
                await self.kg.add_edge(edge)
                count += 1

            except Exception as e:
                log.warning(f"Failed to process causal observation: {e}")

        if count > 0:
            log.debug(f"Processed {count} causal observations into graph")

        # Phase 19: Wire GeneralizationEngine — attempt principle extraction
        # when we have enough observations to generalize from
        if count >= 3 and self.generalization_engine:
            try:
                principle = await self.generalization_engine.abstract_principles(observations)
                if principle:
                    log.info(f"📐 New principle discovered: '{principle.name}'")
            except Exception as e:
                log.warning(f"Principle extraction failed (non-fatal): {e}")

        return count

    async def _process_contradictions(
        self, contradictions: list[Contradiction]
    ) -> int:
        """Process contradictions detected by Gemini."""
        count = 0
        for c in contradictions:
            try:
                result = await self.contradictions.process_contradiction(c)
                if result:
                    count += 1
            except Exception as e:
                log.warning(f"Failed to process contradiction: {e}")

        if count > 0:
            log.debug(f"Processed {count} contradictions")
        return count

    async def _process_hypotheses(self, hypotheses: list[Hypothesis]) -> int:
        """Store hypotheses generated by Gemini."""
        count = 0
        for h in hypotheses:
            try:
                await self.hypotheses.store_hypothesis(h)
                count += 1
            except Exception as e:
                log.warning(f"Failed to store hypothesis: {e}")

        if count > 0:
            log.debug(f"Stored {count} hypotheses")
        return count

    async def _process_hypothesis_resolutions(
        self, resolutions: list[HypothesisResolution]
    ) -> int:
        """Apply confirm/deny for pending hypotheses (structured model output)."""
        if not resolutions:
            return 0
        count = 0
        for hr in resolutions:
            try:
                stored = await self.hypotheses.get_by_id(hr.hypothesis_id)
                if not stored:
                    log.warning(
                        f"Hypothesis resolution skipped — unknown id {hr.hypothesis_id[:8]}..."
                    )
                    continue
                if stored.status != "pending":
                    log.debug(
                        f"Hypothesis {hr.hypothesis_id[:8]} already {stored.status}, skipping resolution"
                    )
                    continue
                if hr.action == "confirm":
                    await self.hypotheses.confirm(hr.hypothesis_id)
                else:
                    await self.hypotheses.deny(hr.hypothesis_id)
                if hr.notes:
                    log.debug(f"Hypothesis {hr.action}: {hr.notes[:80]}")
                count += 1
            except Exception as e:
                log.warning(f"Failed to process hypothesis resolution: {e}")
        if count > 0:
            log.debug(f"Resolved {count} hypotheses")
        return count

    async def _process_uncertainty_tracking(
        self, entries: list[UncertaintyTrackingEntry]
    ) -> int:
        """Persist structured uncertainty notes via DebateEngine (uncertainties table)."""
        if not entries:
            return 0
        count = 0
        for e in entries:
            topic = (e.topic or "").strip()
            why = (e.why_uncertain or "").strip()
            if not topic or not why:
                log.debug("Skipping uncertainty_tracking entry with empty topic or reason")
                continue
            try:
                await self.debate_engine.track_uncertainty(topic, why)
                count += 1
            except Exception as ex:
                log.warning(f"Failed to record uncertainty: {ex}")
        if count > 0:
            log.debug(f"Recorded {count} uncertainty topics")
        return count

    # ------------------------------------------------------------------
    # UI Command Handlers
    # ------------------------------------------------------------------

    async def get_all_memories(self) -> list[Memory]:
        return await self.memory.all_memories()

    async def get_active_goals(self):
        return await self.goals.get_active()

    async def get_all_goals(self):
        return await self.goals.get_all()

    async def search_memories(self, query: str) -> list[Memory]:
        return await self.memory.search(query)

    async def add_manual_memory(self, content: str) -> Memory:
        return await self.memory.add_manual(content)

    async def forget_memory(self, index: int) -> bool:
        return await self.memory.delete_by_index(index)

    async def archive_memory(self, memory_id: str) -> bool:
        return await self.memory.archive(memory_id)

    async def update_memory_confidence(self, memory_id: str, confidence: float) -> bool:
        return await self.memory.update_confidence(memory_id, confidence)

    async def merge_memories(self, keep_id: str, merge_id: str) -> bool:
        return await self.memory.merge(keep_id, merge_id)

    async def get_all_sessions(self) -> list[Session]:
        return await self.session.get_all_sessions()

    # Phase 2 command handlers

    async def get_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        return await self.kg.stats()

    async def get_graph_neighborhood(self, query: str) -> dict | None:
        """Get the neighborhood of a concept in the graph."""
        node_id = self.kg.find_node_by_content(query)
        if not node_id:
            return None
        return await self.kg.get_neighborhood(node_id, depth=2)

    async def get_causal_chain(self, from_concept: str, to_concept: str) -> list[dict] | None:
        """Find causal chain between two concepts."""
        src = self.kg.find_node_by_content(from_concept)
        tgt = self.kg.find_node_by_content(to_concept)
        if not src or not tgt:
            return None
        return await self.kg.find_causal_chain(src, tgt)

    async def get_contradictions(self) -> list[StoredContradiction]:
        return await self.contradictions.get_unresolved()

    async def get_all_contradictions(self) -> list[StoredContradiction]:
        return await self.contradictions.get_all()

    async def get_hypotheses(self) -> list[StoredHypothesis]:
        return await self.hypotheses.get_pending()

    async def get_all_hypotheses(self) -> list[StoredHypothesis]:
        return await self.hypotheses.get_all()

    async def resolve_hypothesis(self, hypothesis_id: str, action: str) -> bool:
        """
        Mark a pending hypothesis confirmed or denied (CLI / operator).

        action must be 'confirm' or 'deny'. Returns True if a row was updated.
        """
        aid = hypothesis_id.strip()
        if action not in ("confirm", "deny"):
            return False
        stored = await self.hypotheses.get_by_id(aid)
        if not stored or stored.status != "pending":
            return False
        if action == "confirm":
            await self.hypotheses.confirm(aid)
        else:
            await self.hypotheses.deny(aid)
        return True

    async def get_recent_improvements(self):
        """Fetch recent self-corrections."""
        return await self.improver.get_recent_improvements()

    async def get_principles(self):
        """Fetch all discovered universal principles."""
        return await self.generalization_engine.get_all_principles()

    async def get_uncertainties(self):
        """Fetch tracked uncertainties."""
        return await self.debate_engine.get_uncertainties()

    async def get_proposals(self):
        """Fetch pending self-improvement proposals."""
        return await self.meta_reasoning.get_pending_proposals()

    async def get_all_proposals(self):
        """Fetch all self-improvement proposals."""
        return await self.meta_reasoning.get_all_proposals()

    async def resolve_improvement_proposal(self, proposal_id: str, status: str) -> bool:
        """
        Operator workflow: approve, reject, or mark implemented. Validates UUID row exists.
        """
        allowed = {"approved", "rejected", "implemented"}
        if status not in allowed:
            return False
        pid = proposal_id.strip()
        row = await self.db.fetch_one(
            "SELECT id FROM improvement_proposals WHERE id = ?",
            (pid,),
        )
        if not row:
            return False
        await self.meta_reasoning.update_status(pid, status)
        return True

    async def run_benchmark(self, status_callback=None):
        """Run the benchmark suite."""
        return await self.benchmark.run(self, status_callback=status_callback)

    async def get_benchmark_history(self):
        """Fetch benchmark history."""
        return await self.benchmark.get_history()

    async def run_meta_analysis(self, status_callback=None):
        """Trigger meta-reasoning analysis."""
        return await self.meta_reasoning.analyze_and_propose(status_callback=status_callback)

    async def run_debate(self, topic: str, status_callback: Callable[..., Any] | None = None):
        """Manually trigger a Phase 4 debate."""
        resolution = await self.debate_engine.run_debate(topic, rounds=1, status_callback=status_callback)
        # Apply the graph updates discovered during the debate
        if resolution.graph_updates:
            await self._process_causal_observations(resolution.graph_updates)
        return resolution

    async def export_session(self) -> str | None:
        """Export the current session's conversation to a JSON file."""
        session = self.session.current
        if not session or session.turn_count == 0:
            return None

        turns = await self.session.get_recent_turns(limit=9999)
        memories = await self.memory.all_memories()
        goals = await self.goals.get_all()
        graph_stats = await self.kg.stats()

        export = {
            "session_id": session.id,
            "started_at": session.started_at,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": session.turn_count,
            "avg_confidence": session.avg_confidence,
            "graph": graph_stats,
            "turns": [
                {
                    "turn": t.turn_number,
                    "user": t.user_input,
                    "reasoning": t.reasoning,
                    "response": t.response,
                    "self_reflection": t.self_reflection,
                    "confidence": t.confidence,
                }
                for t in turns
            ],
            "memories": [
                {
                    "content": m.content,
                    "importance": m.importance,
                    "source": m.source.value if hasattr(m.source, 'value') else str(m.source),
                    "tags": m.tags,
                }
                for m in memories
            ],
            "goals": [
                {
                    "description": g.description,
                    "status": g.status.value if hasattr(g.status, 'value') else str(g.status),
                    "priority": g.priority.value if hasattr(g.priority, 'value') else str(g.priority),
                }
                for g in goals
            ],
        }

        export_dir = VYN_EXPORTS
        export_dir.mkdir(exist_ok=True)

        filename = f"aria_session_{session.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = export_dir / filename
        filepath.write_text(json.dumps(export, indent=2), encoding="utf-8")
        return str(filepath)

    async def get_session_info(self) -> dict:
        """Get current session information including graph stats."""
        session = self.session.current
        total_memories = await self.memory.count()
        active_goals = await self.goals.count_active()
        total_turns = await self.session.get_total_turns()
        graph_stats = await self.kg.stats()

        return {
            "session_id": session.id[:8] if session else "none",
            "turn_count": session.turn_count if session else 0,
            "total_turns": total_turns,
            "total_memories": total_memories,
            "active_goals": active_goals,
            "memories_this_session": session.memories_created if session else 0,
            "avg_confidence": session.avg_confidence if session else 0.0,
            "graph_nodes": graph_stats["total_nodes"],
            "graph_edges": graph_stats["total_edges"],
        }

    async def get_health_status(self) -> dict:
        """Operator-facing runtime health and capability status."""
        provider_settings = get_provider_settings(self.settings_store)
        return {
            "database_path": str(self.db.db_path),
            "data_dir": str(VYN_HOME),
            "project_root": str(WORKSPACE_DIR),
            "vector_store_active": bool(getattr(self.vector_store, "client", None)),
            "execution_backend": getattr(getattr(self.tool_registry.tools.get("run_terminal_command"), "environment", None), "name", "none"),
            "browser_registered": "browser" in self.tool_registry.tools,
            "current_session": self.session.current.id if self.session.current else None,
            "autonomy_policy": autonomy_policy_snapshot(),
            "provider": provider_settings["provider"],
            "model": provider_settings["model"],
            "telegram_public_mode": telegram_public_mode_enabled(),
            "setup": self.settings_store.setup_status(),
        }

    async def get_setup_status(self) -> dict:
        return self.settings_store.setup_status()

    async def get_runtime_settings(self) -> dict:
        settings = self.settings_store.load_settings()
        status = self.settings_store.setup_status()
        return {
            "settings": settings,
            "status": status,
            "providers": list_providers(),
        }

    async def update_runtime_settings(self, payload: dict[str, Any]) -> dict:
        saved = self.settings_store.save_settings(payload)
        if "web_api_key" in payload:
            self.settings_store.set_web_api_key(str(payload["web_api_key"]))
        provider_secrets = payload.get("provider_secrets", {})
        for provider, secret in provider_secrets.items():
            if secret:
                self.settings_store.set_provider_secret(provider, str(secret))
        self.reload_provider()
        return saved

    async def get_usage_summary(self) -> dict:
        return await self.usage_tracker.summary()

    async def list_supported_providers(self) -> list[dict]:
        return list_providers()

    async def test_provider_credentials(
        self, *, provider: str, api_key: str, model: str | None = None
    ) -> dict[str, Any]:
        """Live connectivity check using a throwaway config (does not overwrite runtime keys)."""
        from silex.llm.provider_test import ping_provider

        return await ping_provider(provider, api_key, model=model)

    def reload_provider(self) -> None:
        provider_settings = get_provider_settings(self.settings_store)
        self.gemini = build_provider(self.settings_store, self.usage_tracker)
        self.gemini.connect()
        self.router = ModelRouter(
            fast_model=provider_settings["fast_model"],
            reasoning_model=provider_settings["reasoning_model"],
        )
        self.pruner = ContextPruner(self.gemini)
        self.context_builder.pruner = self.pruner
        self.generalization_engine = GeneralizationEngine(self.gemini, self.db)
        self.context_builder.generalization_engine = self.generalization_engine
        self.critic = ResponseCritic(self.gemini)
        self.debate_engine = DebateEngine(self.gemini, self.db)
        self.meta_reasoning = MetaReasoningEngine(self.gemini, self.db)
        self.benchmark = BenchmarkRunner(self.gemini, self.db)

    def _acquire_process_lock(self) -> None:
        if allow_multi_writer():
            return
        role = get_process_role()
        if self._process_lock_path.exists():
            try:
                existing = self._process_lock_path.read_text(encoding="utf-8").strip()
                if existing:
                    lock_data = json.loads(existing)
                    pid = lock_data.get("pid")
                    if pid:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            self._process_lock_path.unlink(missing_ok=True)
                            log.warning("Stale process lock cleaned up at %s for pid %s", self._process_lock_path, pid)
                        except OSError as exc:
                            # Windows: WinError 87 = Invalid Parameter (Not running), WinError 11 = Access Denied.
                            # We treat both (and Unix ESRCH) as stale lock triggers for recovery.
                            win_err = getattr(exc, "winerror", None)
                            if exc.errno == errno.ESRCH or win_err == 87 or win_err == 11:
                                self._process_lock_path.unlink(missing_ok=True)
                                log.warning("Stale process lock cleaned up at %s for pid %s (err %s)", self._process_lock_path, pid, win_err or exc.errno)
                            else:
                                raise RuntimeError(f"LOCK_EXISTS:{pid}") from exc
                        else:
                            raise RuntimeError(f"LOCK_EXISTS:{pid}")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        self._process_lock_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "role": role,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )

    def _release_process_lock(self) -> None:
        if allow_multi_writer():
            return
        try:
            if self._process_lock_path.exists():
                self._process_lock_path.unlink()
        except OSError:
            pass
