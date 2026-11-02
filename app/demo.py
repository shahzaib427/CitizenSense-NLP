"""CLI demo.  Run from the project root:  python app/demo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.utils import analyze_feedback  # noqa: E402


def show(r):
    if "error" in r:
        print(r["error"]); return
    print("\n" + "-" * 60)
    print(f"Category            : {r['category']}  (confidence {r['confidence']['category']:.0%})")
    print(f"Sentiment           : {r['sentiment']}")
    print(f"Priority            : {r['priority']}")
    print(f"Keywords            : {', '.join(r['keywords'])}")
    print(f"Suggested Department: {r['suggested_department']}")
    print(f"Suggested Action    : {r['suggested_action']}")
    print(f"Summary             : {r['summary']}")
    if r["needs_human_review"]:
        print("!! Low confidence - send to HUMAN REVIEW queue")
    print("-" * 60)
    print("(Prototype output: predictions are suggestions, not final decisions.)\n")


if __name__ == "__main__":
    print("CitizenSense CLI - type 'quit' to exit")
    while True:
        text = input("Enter feedback text: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            show(analyze_feedback(text))
