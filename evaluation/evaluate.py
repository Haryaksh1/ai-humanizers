import sys
from pathlib import Path

# Ensure project root on path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import score_text  # noqa: E402


def demo():
    balanced_text = (
        "This tool converts formal, robotic writing into clear, natural prose. "
        "It keeps the original meaning while improving flow and readability."
    )

    aggressive_text = (
        "Imagine your paragraphs suddenly breathing—phrases loosen up, tone warms, "
        "and stiff constructs crack into lively, varied sentences!"
    )

    print("== Balanced output ==")
    print(balanced_text)
    print(score_text(balanced_text))
    print()

    print("== Aggressive output ==")
    print(aggressive_text)
    print(score_text(aggressive_text))


if __name__ == "__main__":
    demo()
