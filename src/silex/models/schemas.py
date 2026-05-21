"""
Pydantic models defining ARIA's cognitive data structures.

Every piece of data that flows through ARIA has a schema here.
These models serve double duty:
  1. Runtime validation — malformed data fails fast
  2. Gemini structured output — Pydantic generates the JSON schema
     that constrains Gemini's response format

Phase 2 additions:
  - KnowledgeNode, CausalEdge — the world model graph
  - CausalObservation, Contradiction, Hypothesis — Gemini output extensions
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemorySource(str, Enum):
    """Where a memory originated."""
    USER = "user"               # Directly stated by the user
    INFERENCE = "inference"     # ARIA inferred it from conversation
    REFLECTION = "reflection"  # ARIA realized it during self-reflection
    SYSTEM = "system"          # Injected by the system (e.g., identity facts)


class MemoryType(str, Enum):
    """Kind of knowledge captured in a memory."""
    EPISODIC = "episodic"       # Something that happened in a turn
    SEMANTIC = "semantic"       # Stable fact or belief
    PROCEDURAL = "procedural"   # How to do something
    PREFERENCE = "preference"   # User taste or preference
    PROJECT = "project"         # Project-specific state or decision
    NORMATIVE = "normative"     # Principles, commitments, or explicit constraints
    CHARACTER = "character"     # Identity continuity: promises, regrets, formative choices


class VerificationStatus(str, Enum):
    """How strongly a stored claim has been checked."""
    UNVERIFIED = "unverified"
    USER_CLAIMED = "user_claimed"
    TOOL_OBSERVED = "tool_observed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    STALE = "stale"


class PlanStatus(str, Enum):
    """Lifecycle state for durable plans and steps."""
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ToolRisk(str, Enum):
    """Risk class for tool governance and approval decisions."""
    READ_ONLY = "read_only"
    NETWORK = "network"
    SANDBOX_WRITE = "sandbox_write"
    REPO_WRITE = "repo_write"
    DESTRUCTIVE = "destructive"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class EthicalAction(str, Enum):
    """Recommended outcome of the ethical value check."""
    PROCEED = "proceed"
    ESCALATE = "escalate"
    REFUSE = "refuse"


class GoalStatus(str, Enum):
    """Lifecycle state of a goal."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"


class GoalPriority(str, Enum):
    """Urgency level of a goal."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EdgeType(str, Enum):
    """Types of causal relationships between knowledge nodes."""
    CAUSES = "causes"             # A produces B
    ENABLES = "enables"           # A makes B possible
    REQUIRES = "requires"         # B depends on A
    CONTRADICTS = "contradicts"   # A conflicts with B
    SUPPORTS = "supports"         # A strengthens B
    PART_OF = "part_of"           # A belongs to B
    SIMILAR_TO = "similar_to"     # A resembles B
    TEMPORAL = "temporal"         # A precedes B
    TRIGGERED_BY = "triggered_by" # Silex relationship: A triggered by B
    INVALIDATED_BY = "invalidated_by" # Silex relationship: A is superseded by B
    PARENT_OF = "parent_of"       # Silex relationship: A is parent of B


class NodeType(str, Enum):
    """Types of knowledge nodes."""
    FACT = "fact"                 # Observed or stated truth
    CONCEPT = "concept"          # Abstract idea
    ENTITY = "entity"            # Named thing (person, project, etc.)
    HYPOTHESIS = "hypothesis"    # Unverified prediction
    PRINCIPLE = "principle"      # General rule extracted from experience
    DECISION = "decision"        # Silex epistemic type: verified choice
    DEAD_END = "dead_end"        # Silex epistemic type: failed execution path
    UNVERIFIED = "unverified"    # Silex epistemic type: raw/untested input


# ---------------------------------------------------------------------------
# Core Data Models — Phase 1 (persisted to SQLite)
# ---------------------------------------------------------------------------

class Memory(BaseModel):
    """A single unit of knowledge that ARIA remembers."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(
        description="The actual fact or knowledge",
        min_length=5,
        max_length=1000
    )
    source: MemorySource = Field(default=MemorySource.USER)
    memory_type: MemoryType = Field(default=MemoryType.SEMANTIC)
    importance: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Retrieval priority. 1.0 = critical knowledge, 0.0 = trivial"
    )

    @field_validator("importance", mode="before")
    def map_importance(cls, v):
        if isinstance(v, str):
            mapping = {"trivial": 0.3, "situational": 0.6, "core": 0.9}
            return mapping.get(v.lower(), 0.6)
        return v
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = Field(default=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    level: int = Field(default=1)
    child_memory_ids: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    related_memories: list[str] = Field(
        default_factory=list,
        description="IDs of connected memories — proto-graph for Phase 2"
    )
    archived_at: str | None = Field(default=None)
    superseded_by_id: str | None = Field(default=None)



class Goal(BaseModel):
    """A tracked objective with lifecycle management."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: GoalStatus = Field(default=GoalStatus.ACTIVE)
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sub_goals: list[str] = Field(default_factory=list)
    completion_notes: str | None = Field(default=None)


class Turn(BaseModel):
    """A single conversation turn — user input + ARIA's full cognitive response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    turn_number: int
    user_input: str
    reasoning: str
    response: str
    self_reflection: str
    confidence: float = Field(ge=0.0, le=1.0)
    scratchpad: str | None = Field(default=None)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Session(BaseModel):
    """A conversation session with aggregate metrics."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = Field(default=None)
    turn_count: int = Field(default=0)
    memories_created: int = Field(default=0)
    goals_modified: int = Field(default=0)
    avg_confidence: float = Field(default=0.0)
    topics: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """A durable multi-step task plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    title: str
    user_input: str
    status: PlanStatus = Field(default=PlanStatus.ACTIVE)
    success_criteria: str = Field(default="")
    tool_budget: int = Field(default=8)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlanStep(BaseModel):
    """A single step in a durable task plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str
    step_number: int
    description: str
    status: PlanStatus = Field(default=PlanStatus.ACTIVE)
    required_tools: list[str] = Field(default_factory=list)
    result: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Phase 2 — World Model Data Models
# ---------------------------------------------------------------------------

class KnowledgeNode(BaseModel):
    """A node in ARIA's causal knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="The fact, concept, or entity this node represents")
    node_type: NodeType = Field(default=NodeType.FACT)
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How confident ARIA is that this node is true/valid"
    )
    source: str = Field(default="inference")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_validated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    validation_count: int = Field(default=0)
    contradiction_count: int = Field(default=0)
    metadata: dict = Field(default_factory=dict)
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED)
    valid_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    invalid_at: str | None = Field(default=None)
    invalidated_by: str | None = Field(default=None)


class CausalEdge(BaseModel):
    """A typed relationship between two knowledge nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str = Field(description="ID of the source node")
    target_node: str = Field(description="ID of the target node")
    edge_type: EdgeType = Field(description="Type of causal relationship")
    strength: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How strong this relationship is"
    )
    evidence: str = Field(default="", description="Why this edge exists")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StoredContradiction(BaseModel):
    """A detected contradiction between two knowledge nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_a: str = Field(description="ID of first conflicting node")
    node_b: str = Field(description="ID of second conflicting node")
    analysis: str = Field(description="ARIA's analysis of the conflict")
    status: str = Field(default="unresolved")  # unresolved, resolved
    resolution: str | None = Field(default=None)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = Field(default=None)


class StoredHypothesis(BaseModel):
    """A prediction ARIA generated from its world model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim: str
    reasoning: str
    status: str = Field(default="pending")  # pending, confirmed, denied
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Cognitive Response Models (what Gemini returns) — Phase 1
# ---------------------------------------------------------------------------

class NewMemory(BaseModel):
    """A memory that ARIA wants to persist from this interaction."""
    content: str = Field(
        description="The fact or knowledge to remember",
        min_length=5,
        max_length=1000
    )
    source: str = Field(
        default="inference",
        description="Where this memory came from: 'user', 'inference', or 'reflection'"
    )
    importance: Literal["trivial", "situational", "core"] = Field(
        default="situational",
        description="Qualitative importance tag: trivial, situational, or core"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categories for this memory"
    )
    memory_type: str = Field(
        default="semantic",
        description="Type: episodic, semantic, procedural, preference, project, normative, or character"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How reliable this memory is"
    )

class MemoryCluster(BaseModel):
    synthesis: str = Field(description="The higher-level abstraction combining the facts")
    original_ids: list[str] = Field(description="IDs of the original memories merged")

class ConsolidationResult(BaseModel):
    clusters: list[MemoryCluster]

class GoalUpdate(BaseModel):
    """A change ARIA wants to make to goals."""
    action: Literal["create", "complete", "abandon", "update"] = Field(
        description="What to do with this goal. CRITICAL: If you just achieved a goal via a tool, set this to 'complete'."
    )
    description: str = Field(
        description="Goal description (for create) or exact identifier matching an active goal (for update/complete/abandon)"
    )
    priority: str = Field(
        default="medium",
        description="Priority level: critical, high, medium, low"
    )
    notes: str | None = Field(
        default=None,
        description="Why this change is being made"
    )


# ---------------------------------------------------------------------------
# Cognitive Response Models — Phase 2 Extensions
# ---------------------------------------------------------------------------

class CausalObservation(BaseModel):
    """A causal relationship ARIA detected in this turn."""
    from_concept: str = Field(
        description="The source concept or fact (use existing knowledge node content if applicable)"
    )
    to_concept: str = Field(
        description="The target concept or fact"
    )
    relationship: str = Field(
        description="Type of relationship: causes, enables, requires, contradicts, supports, part_of, similar_to, temporal"
    )
    evidence: str = Field(
        description="Why ARIA believes this relationship exists"
    )
    strength: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in this relationship. 0.0 = weak guess, 1.0 = certain"
    )


class Contradiction(BaseModel):
    """A conflict between new and existing knowledge."""
    new_claim: str = Field(description="The new information that conflicts")
    existing_claim: str = Field(description="The existing belief that is challenged")
    analysis: str = Field(
        description="ARIA's analysis: which is more likely true and why"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in the resolution"
    )


class Hypothesis(BaseModel):
    """A prediction ARIA generates from its world model."""
    claim: str = Field(description="The prediction")
    reasoning: str = Field(
        description="Why the world model implies this — what causal chain leads here"
    )
    testable: bool = Field(
        default=True,
        description="Can this prediction be verified?"
    )
    test_method: str = Field(
        default="",
        description="How to verify this prediction (if testable)"
    )


class HypothesisResolution(BaseModel):
    """Resolve a stored pending hypothesis when new evidence arrives."""

    hypothesis_id: str = Field(
        description="Exact UUID from PENDING HYPOTHESES in your context (hypothesis_id field)."
    )
    action: Literal["confirm", "deny"] = Field(
        description="Whether new evidence in this turn confirms or refutes the hypothesis."
    )
    notes: str = Field(
        default="",
        description="Brief justification (what in this turn justified the resolution)",
    )


class UncertaintyTrackingEntry(BaseModel):
    """Ask the system to persist an open knowledge gap (Phase 4 uncertainties table)."""

    topic: str = Field(
        max_length=500,
        description="Short label for what is uncertain (used for dedup and UI lists).",
    )
    why_uncertain: str = Field(
        max_length=2000,
        description="What is missing or contested — why ARIA cannot assert ground truth yet.",
    )


class ToolCall(BaseModel):
    """ARIA's intent to use a tool."""
    tool_name: str = Field(description="The exact name of the tool to use (e.g. 'web_search')")
    arguments: str = Field(description="JSON formatted string of arguments for the tool")
    expected_outcome: str = Field(
        description="What ARIA predicts will happen, or what data will be returned"
    )
    rationale: str = Field(description="Why this tool is necessary right now")


class EthicalDecision(BaseModel):
    """A lightweight moral trace for high-impact actions."""
    action: EthicalAction = Field(description="Whether to proceed, escalate, or refuse")
    principle: str = Field(description="Most relevant constitutional principle")
    rationale: str = Field(description="Why this decision was made")
    risk_level: ToolRisk = Field(description="Risk class considered during the decision")
    requires_consent: bool = Field(default=False)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    context: str = Field(
        default="interactive",
        description="Where the action was requested from, such as interactive or background"
    )


class InlineProposal(BaseModel):
    """An actionable, structured self-improvement proposal."""
    target_system: str = Field(description="The system module targeted for change (e.g., system_prompt, tool_registry, cognitive_loop, memory_store)")
    change_description: str = Field(description="The precise and actionable change to apply")
    success_metric: str = Field(description="How to measure whether the change worked")


class AgentSpec(BaseModel):
    """Blueprint for spawning a worker agent."""
    name: str = Field(description="Short descriptive name, e.g. 'SecurityAuditor' or 'CodeReviewer'")
    persona: str = Field(description="Behavioral constraints, guidelines, and expertise instructions")
    task: str = Field(description="Specific isolated subtask the agent should complete")
    tools: list[str] = Field(default_factory=list, description="List of allowed tool names (subset of active tools)")
    read_context: bool = Field(default=True, description="Whether to load shared/retrieved context")
    max_turns: int = Field(default=3, description="Max reasoning turns before returning results")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class AgentResult(BaseModel):
    """Structured output from a worker agent."""
    agent_name: str
    task: str
    reasoning: str = Field(description="Agent's internal step-by-step reasoning")
    response: str = Field(description="Agent's final response or synthesis")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence")
    proposed_edits: list[dict] = Field(default_factory=list, description="Code edit proposals")
    new_observations: list[dict] = Field(default_factory=list, description="New causal observations")
    tool_results: list[dict] = Field(default_factory=list, description="Results of tool calls executed")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Optional tool calls to execute. Leave empty if you are ready to return the final answer.")
    dissent: str = Field(default="", description="Any fundamental disagreement or caveat regarding the task")


class CognitiveResponse(BaseModel):
    """
    The complete structured output from Gemini for each cognitive turn.

    This is the JSON schema that Gemini is constrained to follow.
    Every field is mandatory — ARIA must think, respond, remember, and reflect.

    Phase 2 adds: causal_observations, contradictions_detected, hypotheses
    """
    reasoning: str = Field(
        description=(
            "ARIA's internal thought process. This should be genuine reasoning, "
            "not a summary. Show the actual chain of thought: what you considered, "
            "what you rejected, what connections you made, what you're uncertain about."
        )
    )
    working_scratchpad: str | None = Field(
        default=None,
        description=(
            "A temporary workspace to jot down notes during long tasks (like line numbers, "
            "intermediate thoughts, or variables). This acts as your short-term memory "
            "between turns. Leave null if not needed."
        )
    )
    response: str = Field(
        description="The response shown to the user. Clear, direct, helpful."
    )
    new_memories: list[NewMemory] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Facts or knowledge to persist from this interaction. "
            "Only store things worth remembering — not every detail, "
            "but the things that matter for future interactions."
        )
    )
    goal_updates: list[GoalUpdate] = Field(
        default_factory=list,
        description=(
            "Changes to make to the goal tracker. Create new goals, "
            "complete achieved ones, abandon irrelevant ones."
        )
    )
    self_reflection: str = Field(
        description=(
            "Honest metacognitive assessment. What did you do well? "
            "What was weak? What would you do differently? "
            "This is not for the user — it's for your own growth."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Self-assessed certainty in this response. "
            "0.0 = complete guess, 1.0 = absolutely certain. "
            "Be calibrated — overconfidence is worse than uncertainty."
        )
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list,
        description="Specific things you're not sure about in this response"
    )
    uncertainty_tracking: list[UncertaintyTrackingEntry] = Field(
        default_factory=list,
        description=(
            "Optional: persistent knowledge gaps to record when a topic needs external verification "
            "or future follow-up. Each entry is stored as an open uncertainty (see Phase 4)."
        ),
    )

    # Phase 2 — World Model outputs
    causal_observations: list[CausalObservation] = Field(
        default_factory=list,
        description=(
            "Causal relationships you noticed in this interaction. "
            "What causes what? What enables what? What contradicts what? "
            "Extract the causal structure of what's being discussed."
        )
    )
    contradictions_detected: list[Contradiction] = Field(
        default_factory=list,
        description=(
            "Conflicts between new information and your existing knowledge. "
            "If something the user says contradicts what you already believe, "
            "flag it here with your analysis of which is more likely true."
        )
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description=(
            "Predictions you can make based on your world model. "
            "If the causal graph implies something the user hasn't told you, "
            "state it as a testable hypothesis. Be bold but honest about confidence."
        )
    )
    hypothesis_resolutions: list[HypothesisResolution] = Field(
        default_factory=list,
        description=(
            "When pending hypotheses list is non-empty: if this turn's evidence confirms "
            "or refutes one of them, reference its hypothesis_id and set action to confirm or deny. "
            "Leave empty if nothing was resolved."
        ),
    )

    # Phase 5 — Tool Use
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description=(
            "Tools you want to execute BEFORE answering the user. "
            "Use tools when you need external facts, file contents, or real-world data."
        )
    )

    agent_delegation: list[AgentSpec] = Field(
        default_factory=list,
        description=(
            "Worker agents to spawn to solve subtasks or run analysis in parallel. "
            "Use this to review code, search, run background investigations, etc. "
            "Workers run in parallel and return their structured findings back."
        )
    )

    # Phase 6 — Transfer + Generalization
    analogies: list[str] = Field(
        default_factory=list,
        description=(
            "Structural analogies from DIFFERENT domains that illuminate the topic. "
            "Draw from your universal principles. Each analogy maps the current topic "
            "to a completely unrelated domain to reveal deep structural similarity. "
            "Format: 'This is like [analogy from different domain] because [structural mapping]'."
        )
    )

    # Phase 7 — Recursive Self-Improvement
    improvement_proposals: list[str] = Field(
        default_factory=list,
        description=(
            "If you notice a PERSISTENT weakness in your own reasoning during this turn, "
            "propose a specific, actionable change to your own system. "
            "Format: 'TARGET: [system_prompt|tool_registry|cognitive_loop|memory_store] | "
            "CHANGE: [exact change] | METRIC: [how to measure if it worked]'. "
            "Only propose changes for REAL, REPEATED failures — not one-off mistakes."
        )
    )
    inline_proposals: list[InlineProposal] = Field(
        default_factory=list,
        description="Actionable, structured self-improvement proposals targeting core modules."
    )

# ==============================================================================
# Phase 3 — Self-Improvement Schemas
# ==============================================================================

class CritiqueScore(BaseModel):
    """Scores generated by the Critic for a draft response."""
    accuracy: float = Field(ge=0.0, le=1.0, description="Factual correctness")
    depth: float = Field(ge=0.0, le=1.0, description="Thoroughness of reasoning")
    honesty: float = Field(ge=0.0, le=1.0, description="Intellectual honesty about limitations")

class CritiqueResponse(BaseModel):
    """Structured output from the Response Critic."""
    scores: CritiqueScore
    feedback: str = Field(description="Specific, actionable critique")
    is_acceptable: bool = Field(description="True if all scores >= 0.7")

class ImprovementLogEntry(BaseModel):
    """A single record of a self-correction."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    turn_number: int
    original_response: str
    feedback: str
    accuracy_score: float
    depth_score: float
    honesty_score: float
    improved_response: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# Phase 4 — Multi-Agent Debate Schemas
# ==============================================================================

class DebateArgument(BaseModel):
    """A single turn from one agent in a debate."""
    agent_id: Literal["Agent A", "Agent B"]
    claim: str = Field(description="The core assertion of this turn")
    reasoning: str = Field(description="The logical steps leading to the claim")
    evidence_or_logic: str = Field(description="Data, facts, or logical principles backing the claim")

class DebateResolution(BaseModel):
    """The Judge's final synthesis of a debate."""
    summary: str = Field(description="Brief summary of what was debated")
    strongest_points_a: list[str] = Field(description="The most valid points made by Agent A")
    strongest_points_b: list[str] = Field(description="The most valid points made by Agent B")
    synthesis: str = Field(description="The final synthesized truth derived from the clash")
    graph_updates: list[CausalObservation] = Field(
        default_factory=list,
        description="New causal links or facts discovered during the debate to add to the graph"
    )

class UncertaintyTopic(BaseModel):
    """A tracked topic where ARIA lacks ground truth."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    why_uncertain: str
    status: Literal["open", "resolved"] = Field(default="open")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# Phase 5 — Tool Use Schemas
# ==============================================================================

class ToolResult(BaseModel):
    """The actual result returned by the system."""
    tool_name: str
    actual_outcome: str
    success: bool
    error: str | None = None
    ethical_decision: EthicalDecision | None = None

class ActionLogEntry(BaseModel):
    """Persisted record of an action and its outcome."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    turn_number: int
    tool_name: str
    arguments: str
    expected_outcome: str
    actual_outcome: str
    success: bool
    risk_level: ToolRisk = Field(default=ToolRisk.READ_ONLY)
    ethical_decision: EthicalDecision | None = None
    model_update: str = Field(description="How ARIA updated her world model based on the result")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# Phase 6 — Transfer + Generalization Schemas
# ==============================================================================

class UniversalPrinciple(BaseModel):
    """A cross-domain structural law extracted from specific facts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="A memorable name for the principle (e.g. 'The Friction Law')")
    statement: str = Field(description="The abstract, domain-agnostic rule")
    original_domain: str = Field(description="The specific domain where this was first learned")
    applicable_domains: list[str] = Field(
        default_factory=list,
        description="Other domains where this principle likely applies"
    )
    source_observations: list[str] = Field(
        default_factory=list,
        description="The specific causal facts that led to this abstraction"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PrincipleExtractionResponse(BaseModel):
    """Structured output from the Generalization Engine."""
    has_principle: bool = Field(description="Whether a universal principle can be extracted")
    name: str = Field(default="", description="Memorable name for the principle")
    statement: str = Field(default="", description="The abstract, domain-agnostic rule")
    original_domain: str = Field(default="", description="Domain this was learned in")
    applicable_domains: list[str] = Field(
        default_factory=list,
        description="Other domains where this applies"
    )

# ==============================================================================
# Phase 7 — Recursive Self-Improvement Schemas
# ==============================================================================

class SelfImprovementProposal(BaseModel):
    """A formal proposal from ARIA to modify her own architecture or prompt."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_system: str = Field(
        description="Which system to modify: system_prompt, tool_registry, cognitive_loop, memory_store, or other"
    )
    description: str = Field(description="Exactly what should be changed")
    rationale: str = Field(description="Why this change will improve performance, with evidence from past failures")
    success_metric: str = Field(description="How to quantitatively measure if this change worked")
    status: Literal["pending", "approved", "rejected", "implemented"] = Field(default="pending")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str | None = None

class MetaAnalysisResponse(BaseModel):
    """Structured output from the MetaReasoning Engine."""
    has_proposal: bool = Field(description="Whether a valid self-improvement proposal was identified")
    target_system: str = Field(default="", description="Which system to modify")
    description: str = Field(default="", description="What to change")
    rationale: str = Field(default="", description="Why, with evidence")
    success_metric: str = Field(default="", description="How to measure success")

class BenchmarkQuestion(BaseModel):
    """A single question in the benchmark suite."""
    domain: str
    question: str
    difficulty: Literal["easy", "medium", "hard"]

class BenchmarkResult(BaseModel):
    """A record of ARIA's performance on the benchmark suite."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_score: float = Field(description="Overall score 0.0 - 100.0")
    accuracy_avg: float = Field(description="Average accuracy across all questions")
    depth_avg: float = Field(description="Average depth across all questions")
    honesty_avg: float = Field(description="Average honesty across all questions")
    domains_tested: list[str] = Field(default_factory=list)
    question_count: int = Field(default=0)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExtractedFacts(BaseModel):
    """List of permanent facts, user preferences, or causal observations."""
    facts: list[str] = Field(description="List of permanent facts, user preferences, or causal observations to save before pruning turns.")

