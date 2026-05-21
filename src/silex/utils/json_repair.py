"""
JSON Repair Interceptor — deterministic utility to fix malformed or truncated
JSON payloads emitted by local models (e.g. Ollama stutters, truncations).
"""

import re
from silex.utils.logger import setup_logger

log = setup_logger("silex.utils.json_repair")

def repair_json(json_str: str) -> str:
    """
    Deterministically heals common JSON syntax issues from local models.
    
    Fixes:
    - Markdown code blocks wrapping the JSON (```json ... ```)
    - Trailing commas before closing braces/brackets
    - Missing closing quotes for truncated strings
    - Unbalanced/unclosed braces/brackets due to truncation
    - Extraneous text prefix/suffix
    """
    if not json_str:
        return "{}"

    # 1. Strip potential markdown wrapping
    stripped = json_str.strip()
    if stripped.startswith("```"):
        # Strip until first newline
        stripped = re.sub(r"^```[a-zA-Z0-9-]*\s*", "", stripped)
        # Strip closing backticks
        stripped = re.sub(r"\s*```$", "", stripped)
    stripped = stripped.strip()

    # 2. Extract JSON payload candidate (from first { or [ to the matching closing character)
    first_brace = stripped.find('{')
    first_bracket = stripped.find('[')
    
    start_idx = -1
    if first_brace != -1 and first_bracket != -1:
        start_idx = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_idx = first_brace
    elif first_bracket != -1:
        start_idx = first_bracket
        
    if start_idx == -1:
        # No JSON structure found, return as is
        return stripped

    # Truncate anything before start_idx
    stripped = stripped[start_idx:]

    # Walk the string to find the matching closing character for the outermost JSON container
    end_idx = -1
    stack = []
    in_str = False
    esc = False
    
    for idx, char in enumerate(stripped):
        if char == '\\' and not esc:
            esc = True
        elif char == '"' and not esc:
            in_str = not in_str
            esc = False
        elif not in_str:
            esc = False
            if char in ('{', '['):
                stack.append(char)
            elif char in ('}', ']'):
                if stack:
                    top = stack[-1]
                    if (char == '}' and top == '{') or (char == ']' and top == '['):
                        stack.pop()
                        if not stack:
                            end_idx = idx
                            break
        else:
            esc = False

    # If the outermost container was closed, discard any trailing prose
    if end_idx != -1:
        stripped = stripped[:end_idx + 1]

    # 3. Clean up trailing commas in objects or lists (e.g., {"a": 1,} -> {"a": 1})
    # Remove trailing commas before closing braces/brackets
    stripped = re.sub(r",\s*([}\]])", r"\1", stripped)

    # 4. Handle unclosed quotes at truncation point
    # We walk the string to track whether we are inside an unescaped double quote.
    inside_string = False
    escape = False
    clean_chars = []
    
    for char in stripped:
        if char == '\\' and not escape:
            escape = True
            clean_chars.append(char)
        elif char == '"' and not escape:
            inside_string = not inside_string
            escape = False
            clean_chars.append(char)
        else:
            escape = False
            clean_chars.append(char)
            
    healed = "".join(clean_chars)
    
    # If the string was left open, close the quote
    if inside_string:
        healed += '"'

    # 4.5 Clean up trailing invalid structures at the end of the JSON string (truncation errors)
    trimmed = healed.rstrip()
    
    # Check if the string ends with a colon, indicating a missing value
    if trimmed.endswith(':'):
        healed = trimmed + ' null'
    # Check if it ends with a comma, indicating a missing element
    elif trimmed.endswith(','):
        healed = trimmed[:-1].rstrip()
    # Check for partially generated boolean or null literals at the end
    else:
        # Match trailing partial literals (true, false, null) preceded by whitespace, colon, or comma
        last_word_match = re.search(r'(?<=[\s:,])([a-zA-Z]+)$', trimmed)
        if last_word_match:
            word = last_word_match.group(1)
            # Check if it's a partial match for true/false/null but not complete
            if "true".startswith(word) and word != "true":
                healed = trimmed[:-len(word)] + "true"
            elif "false".startswith(word) and word != "false":
                healed = trimmed[:-len(word)] + "false"
            elif "null".startswith(word) and word != "null":
                healed = trimmed[:-len(word)] + "null"
        # Check if it's a truncated number ending in a dot (e.g. 12.)
        elif re.search(r'\b\d+\.$', trimmed):
            healed = trimmed + '0'

    # 5. Fix unbalanced / unclosed brackets and braces
    # We track braces and brackets using a stack.
    stack = []
    in_str = False
    esc = False
    
    for char in healed:
        if char == '\\' and not esc:
            esc = True
        elif char == '"' and not esc:
            in_str = not in_str
            esc = False
        elif not in_str:
            esc = False
            if char in ('{', '['):
                stack.append(char)
            elif char in ('}', ']'):
                if stack:
                    # Match if possible, otherwise we keep going
                    top = stack[-1]
                    if (char == '}' and top == '{') or (char == ']' and top == '['):
                        stack.pop()
        else:
            esc = False

    # Close any remaining unclosed braces/brackets in reverse order
    for unclosed in reversed(stack):
        if unclosed == '{':
            healed += '}'
        elif unclosed == '[':
            healed += ']'

    # 6. Final safety sweep of trailing commas that could have been created
    healed = re.sub(r",\s*([}\]])", r"\1", healed)

    return healed
