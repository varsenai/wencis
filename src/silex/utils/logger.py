"""
Structured logging for ARIA.

All log output goes through Rich for consistent formatting.
Security: show_locals is disabled to prevent API keys or credentials
from leaking into traceback output.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logger(name: str = "aria", level: str = "INFO") -> logging.Logger:
    """Create a Rich-formatted logger."""
    import sys
    
    # Fix Windows charmap encode errors when printing emojis (e.g. ❌)
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
            # SECURITY: Never show locals in tracebacks — they can contain
            # API keys, tokens, and other secrets from the genai.Client object.
            tracebacks_show_locals=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
