from typing import Dict

try:
    import language_tool_python  # type: ignore
except Exception:  # pragma: no cover
    language_tool_python = None  # type: ignore

from textstat import textstat
import re


def _heuristic_grammar_issues(text: str) -> int:
    """Lightweight fallback if language_tool_python is unavailable.
    Heuristics: missing sentence punctuation, repeated spaces, lowercase 'i' pronoun, excessive punctuation.
    """
    issues = 0
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    # Missing terminal punctuation for lines/sentences
    for s in sentences:
        if s and s[-1] not in ".!?":
            issues += 1
    # Repeated spaces
    issues += len(re.findall(r"\s{2,}", text))
    # Lowercase ' i ' as pronoun
    issues += len(re.findall(r"(^|\s)i(\s|$)", text))
    # Excessive punctuation sequences
    issues += len(re.findall(r"[!?.,]{3,}", text))
    return issues


def score_text(text: str) -> Dict[str, float]:
    """
    Compute readability and grammar diagnostics for a given text.

    Returns a dict with:
      - readability_score (float): Flesch Reading Ease (higher is easier, ~0-100+)
      - grammar_issues_count (int): number of grammar/style issues detected
      - fluency_score (float, optional): 0-1 simple composite of readability and grammar cleanliness
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Readability via Textstat (Flesch Reading Ease)
    try:
        readability = float(textstat.flesch_reading_ease(text or ""))
    except Exception:
        readability = 0.0

    # Grammar via LanguageTool if available, else fallback
    grammar_issues = 0
    if language_tool_python is not None:
        try:
            tool = language_tool_python.LanguageToolPublicAPI("en-US")
            matches = tool.check(text or "")
            grammar_issues = int(len(matches))
        except Exception:
            grammar_issues = _heuristic_grammar_issues(text or "")
    else:
        grammar_issues = _heuristic_grammar_issues(text or "")

    # Simple fluency: blend normalized readability with issues per 100 words
    words = re.findall(r"\b\w+\b", text or "")
    word_count = max(1, len(words))
    issues_per_100 = grammar_issues / word_count * 100.0

    # Normalize readability to 0..1 (FRE typically 0..100+, clip)
    norm_readability = max(0.0, min(1.0, readability / 100.0))
    # Penalize fluency by issues density
    issues_component = max(0.0, 1.0 - (issues_per_100 * 0.5) / 10.0)  # 0.5 penalty per 10 issues/100w
    fluency = round(0.5 * norm_readability + 0.5 * issues_component, 4)

    return {
        "readability_score": float(round(readability, 4)),
        "grammar_issues_count": int(grammar_issues),
        "fluency_score": float(fluency),
    }
