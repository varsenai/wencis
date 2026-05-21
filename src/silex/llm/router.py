"""
Provider-aware LLM router.
"""

from silex.utils.logger import setup_logger

log = setup_logger("silex.llm.router")

class ModelRouter:
    """
    Classifies intent and routes requests to the appropriate fast or reasoning model.
    """

    def __init__(self, fast_model: str, reasoning_model: str):
        self.fast = fast_model
        self.reasoning = reasoning_model

    def route(self, user_input: str, context_size: int = 0) -> str:
        """
        Determines which model to use based on input complexity.
        """
        user_input_lower = user_input.lower()

        reasoning_signals = [
            "architect", "refactor", "debug", "deep dive", "analyze", 
            "complex", "plan", "strategy", "why", "logic", "optimize",
            "recursive", "generalize"
        ]

        # Simple/Fast Signals
        flash_signals = [
            "list", "show", "read", "read file", "what is", "where is",
            "hello", "hi", "status"
        ]

        # 1. Size-based routing (Huge context needs Pro's stability)
        if context_size > 150000:
            log.info("Routing to reasoning model: large context detected.")
            return self.reasoning

        if any(sig in user_input_lower for sig in reasoning_signals):
            log.info("Routing to reasoning model: complexity signal detected.")
            return self.reasoning

        if any(sig in user_input_lower for sig in flash_signals):
            log.info("Routing to fast model: utility signal detected.")
            return self.fast

        log.info("Routing to fast model: default path.")
        return self.fast
