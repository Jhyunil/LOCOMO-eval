#!/usr/bin/env python3
"""
drop_context.py

FINAL_RESULTS 구조로 생성된 JSON에서 모든 dict의 "context" 필드만 제거하고 저장합니다.

Usage:
    python drop_context.py input.json output.json
    # 필드명을 커스터마이즈하려면:
    python drop_context.py input.json output.json --field context
"""

import json
import argparse
from pathlib import Path
from typing import Any


def strip_field(obj: Any, field: str) -> Any:
    """재귀적으로 모든 dict에서 지정 field 키 제거."""
    if isinstance(obj, dict):
        return {k: strip_field(v, field) for k, v in obj.items() if k != field}
    if isinstance(obj, list):
        return [strip_field(v, field) for v in obj]
    return obj


def main():
    p = argparse.ArgumentParser(description="Remove a field (default: context) from nested JSON dicts.")
    p.add_argument("input_json", type=Path)
    p.add_argument("output_json", type=Path)
    p.add_argument("--field", default="context", help="제거할 필드명 (기본: context)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--indent", type=int, default=2, help="출력 들여쓰기 폭 (기본: 2)")
    group.add_argument("--no-indent", action="store_true", help="minified 저장")
    args = p.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = strip_field(data, args.field)

    if args.no_indent:
        indent = None
        separators = (",", ":")
    else:
        indent = args.indent
        separators = None

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=indent, separators=separators)

    print(f"Saved cleaned JSON (without '{args.field}') to: {args.output_json}")


if __name__ == "__main__":
    main()
