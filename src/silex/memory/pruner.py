"""
Context Pruner — ARIA's "Metabolic Optimizer."

Summarizes old conversation history into Knowledge Graph nodes 
to keep the active context window lean, fast, and cost-effective.
"""

from typing import List

from silex.llm.base import SupportsLLM
from silex.models.schemas import Turn, ConsolidationResult
from silex.utils.config import get_provider_settings
from silex.utils.logger import setup_logger

log = setup_logger("silex.memory.pruner")

class ContextPruner:
    """
    Analyzes turn history and compresses old context into high-density summaries.
    """

    def __init__(self, llm: SupportsLLM):
        self.llm = llm

    async def prune(self, turns: List[Turn], session_manager=None, memory_store=None, threshold: int = 10) -> List[Turn]:
        """
        If the number of turns exceeds the threshold, compresses the oldest 20%.
        Returns a pruned/summarized list of turns.
        """
        if len(turns) <= threshold:
            return turns

        num_to_prune = max(1, len(turns) // 5)
        to_prune = turns[:num_to_prune]
        remaining = turns[num_to_prune:]

        log.info(f"Metabolic event triggered: Pruning {num_to_prune} old turns.")

        turns_text = "\n\n".join(
            f"USER: {turn.user_input}\nARIA: {turn.response}" for turn in to_prune
        )

        # Pre-compaction flush
        if memory_store:
            try:
                from silex.models.schemas import ExtractedFacts
                extraction_prompt = (
                    "You are ARIA's Memory Extractor. Below is a list of conversation turns that are about to be pruned. "
                    "Review these conversation turns. Extract any permanent facts, user preferences, or causal observations (A causes B) "
                    "that should be saved permanently.\n"
                    "Return a JSON list of strings. Do not include markdown tags. Output ONLY a valid JSON list conforming to the schema."
                )
                
                result = await self.llm.complete_json(
                    schema=ExtractedFacts,
                    system_prompt=extraction_prompt,
                    user_input=f"Extract facts from these turns:\n\n{turns_text}",
                    model_override=get_provider_settings()["fast_model"]
                )
                
                if result and result.facts:
                    for fact in result.facts:
                        await memory_store.add_manual(content=fact, importance=0.6)
                    log.info(f"💾 Pre-Compaction Flush: Saved {len(result.facts)} permanent memories before turn pruning.")
            except Exception as e:
                log.error(f"Pre-compaction flush failed: {e}")

        # Build a compression prompt
        compression_prompt = (
            "You are ARIA's Metabolic Optimizer. Below is a list of old conversation turns. "
            "Compress them into a single high-density summary that preserves all key facts, "
            "decisions, and causal connections. Output ONLY the summary text."
        )

        try:
            # Use Flash for compression to keep it cheap
            summary_response = await self.llm.think(
                system_prompt=compression_prompt,
                user_input=f"Compress these turns:\n\n{turns_text}",
                model_override=get_provider_settings()["fast_model"],
            )
            
            summary_text = summary_response.response

            # Create a new "Virtual Turn" that holds the summary
            virtual_turn = Turn(
                session_id=to_prune[0].session_id if to_prune else "system",
                turn_number=remaining[0].turn_number - 1 if remaining else 0,
                user_input="[SYSTEM: Context Compression Event]",
                reasoning="Pruned context",
                response=f"Summary of previous {num_to_prune} turns: {summary_text}",
                self_reflection="",
                confidence=1.0
            )

            if session_manager:
                old_ids = [t.id for t in to_prune]
                await session_manager.compress_turns(virtual_turn.session_id, old_ids, virtual_turn)

            return [virtual_turn] + remaining

        except Exception as e:
            log.error(f"Context pruning failed: {e}")
            return turns # Return original if compression fails

    async def consolidate_memories(self, memory_store):
        """Weekly background task to cluster and consolidate redundant memories."""
        memories = await memory_store.db.fetch_all(
            "SELECT * FROM memories WHERE archived_at IS NULL AND level = 1 AND memory_type IN ('semantic', 'procedural') ORDER BY created_at ASC LIMIT 100"
        )
        if len(memories) < 20:
            return
            
        mem_objs = [memory_store._row_to_memory(dict(r)) for r in memories]
        prompt = (
            "You are ARIA's Memory Consolidator. "
            "Review the following active Level-1 memories. Group highly similar or redundant facts "
            "into higher-level abstract concepts. "
            "Return a list of clusters with the 'synthesis' (the new Level-2 abstract memory) "
            "and the 'original_ids' of the Level-1 memories that were merged. "
            "Only cluster things that mean the same thing or are granular details of the same pattern. "
            "Leave distinct, unconnected facts alone. "
            "If a memory is highly specific (like a password, API key, exact file path, or secret), DO NOT cluster it. Only cluster conceptual patterns."
        )
        
        mem_text = "\n".join([f"[{m.id}] {m.content}" for m in mem_objs])
        
        try:
            result = await self.llm.complete_json(
                schema=ConsolidationResult,
                system_prompt=prompt,
                user_input=f"Memories to cluster:\n\n{mem_text}",
                model_override=get_provider_settings()["reasoning_model"]
            )
            
            count = 0
            async with memory_store.db.transaction():
                for cluster in result.clusters:
                    if len(cluster.original_ids) > 1:
                        new_mem = await memory_store.add_manual(
                            content=cluster.synthesis,
                            importance=0.9,
                            level=2,
                            child_memory_ids=cluster.original_ids
                        )
                        if new_mem is not None:
                            count += 1
                            for old_id in cluster.original_ids:
                                if old_id != new_mem.id:
                                    await memory_store.archive(old_id)
                            
            log.info(f"Consolidated {sum(len(c.original_ids) for c in result.clusters if len(c.original_ids) > 1)} memories into {count} abstractions.")
        except Exception as e:
            log.error(f"Memory consolidation failed: {e}")
