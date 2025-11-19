#!/usr/bin/env python3
"""
Count how many QA items each conversation has
in a LoCoMo dataset file (locomo10.json).
"""

import json
from pathlib import Path

# --- configuration ----------------------------------------------------------
JSON_PATH = Path("locomo10.json")   # ← 파일 경로를 필요에 맞게 수정하세요.
# ---------------------------------------------------------------------------


def count_qa_per_conversation(file_path: Path) -> dict[int, int]:
    """
    Parse the LoCoMo dataset and return a dict mapping
    conversation index → number of QA pairs.
    """
    with file_path.open(encoding="utf-8") as fp:
        conversations = json.load(fp)

    # 결과를 {index: count} 형태로 정리
    return {
        idx + 1: len(conv.get("qa", []))
        for idx, conv in enumerate(conversations)
    }


def main() -> None:
    counts = count_qa_per_conversation(JSON_PATH)

    # 예쁘게 출력
    print("Conversation → #QA")
    for cid, n in counts.items():
        print(f"{cid:>2} → {n:>3}")
    print("-" * 18)
    print(f"Total QAs: {sum(counts.values())}")


if __name__ == "__main__":
    main()
