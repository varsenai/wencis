import html
import re

def sanitize_for_injection(text: str | None) -> str:
    """
    Sanitize untrusted text before injecting it into the system prompt.
    1. Removes control characters that could corrupt parsing.
    2. HTML-escapes the string to neutralize XML/HTML boundary breakouts.
    """
    if not text:
        return ""
        
    # Strip control characters except standard whitespace
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Escape HTML/XML entities (<, >, &, ", ')
    sanitized = html.escape(sanitized, quote=True)
    
    return sanitized
