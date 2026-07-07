# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import re

_MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def repair_json(raw: str) -> str:
    """
    Extract clean JSON from LLM output that may include markdown fences,
    leading prose, trailing text, or other formatting artifacts.

    Steps:
    1. Strip ```json ... ``` or ``` ... ``` markdown fences
    2. If string doesn't start with { or [, find the first one
    3. If string doesn't end with } or ], find the last one

    Returns cleaned string. Caller still must json.loads() or pydantic validate.
    """
    text = raw.strip()

    # Step 1: strip markdown fences
    match = _MARKDOWN_JSON_RE.search(text)
    if match:
        text = match.group(1).strip()

    # Step 2: find opening brace/bracket
    if text and text[0] not in ("{", "["):
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                text = text[i:]
                break

    # Step 3: find closing brace/bracket
    if text and text[-1] not in ("}", "]"):
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ("}", "]"):
                text = text[: i + 1]
                break

    return text
