from __future__ import annotations

import re


def sanitize_tool_text(text: str, *, max_chars: int = 700) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"`{1,3}", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*#>]+\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*", "", cleaned)
    cleaned = cleaned.replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) <= max_chars:
        return cleaned

    snippet = cleaned[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not snippet:
        snippet = cleaned[:max_chars].rstrip(" ,;:-")
    return f"{snippet}..."
