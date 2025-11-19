#!/usr/bin/env python3
"""
Concat LoCoMo‑style conversations and rename 'John' everywhere
(speaker field AND in‑text occurrences).

Usage:
  python concat_conversations.py \
      --input  /mnt/data/locomo10_qa_no_cat5.json \
      --output /mnt/data/locomo10_qa_no_cat5_concat.json \
      [--john-name NewName] \
      [--john-conv-ids 1,2,4,6]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

DEFAULT_JOHN_CONV_IDS = {1, 2, 4, 6}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def gather_speakers(data: Dict[str, Any]) -> Set[str]:
    """Collect every distinct speaker name in the dataset."""
    speakers = set()
    for bundle in data.values():
        for msg in bundle.get("conversation", []):
            sp = msg.get("speaker")
            if isinstance(sp, str):
                speakers.add(sp)
    return speakers


def pick_unique(base: str, existing: Set[str]) -> str:
    """Return a name not present in existing, appending _N if needed."""
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def merge_and_rename(
    data: Dict[str, Any],
    john_conv_ids: Set[int],
    john_repl: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Concat conversations and rename John (speaker + in‑text)."""
    merged: List[Dict[str, Any]] = []
    replaced_text_cnt = 0

    # Pattern for in‑text replacement: whole word John
    john_pat = re.compile(r"\bJohn\b")

    # numeric keys only, ascending
    for k in sorted((int(k) for k in data if k.isdigit())):
        conv = data[str(k)].get("conversation", [])
        for msg in conv:
            new_msg = dict(msg)  # shallow copy
            # 1) Rename speaker field when required
            if k in john_conv_ids and new_msg.get("speaker") == "John":
                new_msg["speaker"] = john_repl

            # 2) Replace in‑text occurrences everywhere
            txt = new_msg.get("text")
            if isinstance(txt, str):
                new_txt, n_sub = john_pat.subn(john_repl, txt)
                if n_sub:
                    replaced_text_cnt += n_sub
                    new_msg["text"] = new_txt

            merged.append(new_msg)

    return merged, replaced_text_cnt


def main() -> None:
    p = argparse.ArgumentParser(description="Concat & rename John everywhere")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--john-name", type=str, default=None,
                   help="Desired replacement for 'John' (default RenamedJohn)")
    p.add_argument("--john-conv-ids", type=str, default="1,2,4,6",
                   help="Comma‑separated conversation IDs containing John speakers")
    args = p.parse_args()

    john_conv_ids = {int(x) for x in args.john_conv_ids.split(",") if x.strip().isdigit()}
    data = load_json(args.input)

    # choose unique replacement name
    existing_speakers = gather_speakers(data)
    john_repl = pick_unique(args.john_name or "RenamedJohn", existing_speakers)

    merged, text_subs = merge_and_rename(data, john_conv_ids, john_repl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"conversation": merged}, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(merged)} messages → {args.output}\n"
        f"Speaker 'John' ⇒ '{john_repl}' in conv IDs {sorted(john_conv_ids)}\n"
        f"In‑text replacements performed: {text_subs}"
    )


if __name__ == "__main__":
    main()
