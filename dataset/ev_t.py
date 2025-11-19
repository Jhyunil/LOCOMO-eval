# -*- coding: utf-8 -*-
"""
LoCoMo 세션 토큰 계산 (conv_idx 자동 탐색 지원)
pip install tiktoken
"""
from pathlib import Path
import json, re, argparse, tiktoken
from typing import List, Dict, Any
import re

ENC = tiktoken.get_encoding("cl100k_base")          # GPT-4(o)/3.5 계열

# ──────────────────────────────────────────────────────────
def load_json(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def join_text(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(t["text"] for t in turns if "text" in t)

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

# ──────────────────────────────────────────────────────────
def find_conv_idx(data: List[Dict[str, Any]], session_key: str) -> int | None:
    """session_key 가 들어 있는 대화 index (없으면 None)"""
    for idx, obj in enumerate(data):
        if session_key in obj["conversation"]:
            return idx
    return None

def count_tokens_for_session(
    data: List[Dict[str, Any]], conv_idx: int, session_key: str
) -> int:
    turns = data[conv_idx]["conversation"][session_key]
    return count_tokens(join_text(turns))

def count_tokens_for_conversation(
    data, conv_idx: int, include_speaker: bool = False
) -> int:
    """
    conversation 하나 전체(=모든 세션) 토큰 수 반환
    - include_speaker=True 이면 "Caroline: ..." 식으로 이름을 붙여 계산
    """
    conv = data[conv_idx]["conversation"]

    text_chunks = []
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")

    # 세션 키만 골라 정렬
    for key in sorted(conv.keys(), key=lambda k: int(re.sub(r"\D", "", k) or -1)):
        if not key.startswith("session_") or "_date_time" in key:
            continue
        for turn in conv[key]:
            if "text" not in turn:
                continue
            # speaker prefix 선택적으로 붙이기
            if include_speaker:
                prefix = f"{turn['speaker']}: "
                text_chunks.append(prefix + turn["text"])
            else:
                text_chunks.append(turn["text"])

    blob = "\n".join(text_chunks)
    return len(ENC.encode(blob))

def conv_stats(data, conv_idx: int) -> dict:
    conv = data[conv_idx]["conversation"]
    turns = [t for k in conv if k.startswith("session_") and "_date" not in k
             for t in conv[k] if "text" in t]

    txt_only   = " ".join(t["text"] for t in turns)
    txt_spk    = " ".join(f"{t['speaker']}: {t['text']}" for t in turns)

    return {
        "turns": len(turns),
        "tok_text": len(ENC.encode(txt_only)),
        "tok_text+spk": len(ENC.encode(txt_spk)),
    }


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoCoMo 특정 세션 토큰 수 계산기",
    )
    parser.add_argument("json_path", help="LoCoMo JSON 경로")
    parser.add_argument("session_key", help="ex) session_27")
    parser.add_argument(
        "-i", "--index",
        type=int,
        default=None,
        help="대화 인덱스(모르면 생략, 자동 탐색)",
    )
    args = parser.parse_args()

    data = load_json(args.json_path)

    # conv_idx 자동 탐색
    conv_idx = args.index
    if conv_idx is None:
        conv_idx = find_conv_idx(data, args.session_key)
        if conv_idx is None:
            raise ValueError(f"'{args.session_key}' not found in any conversation.")
        print(f"[info] '{args.session_key}' found in conversation #{conv_idx}")

    n_tok = count_tokens_for_session(data, conv_idx, args.session_key)
    print(f"conversation[{conv_idx}] / {args.session_key} → {n_tok:,} tokens")


    sum_total = 0
    for i in range(10) :
        conv_idx = i
        info = conv_stats(data, conv_idx)
        print(info)
        
        total_tok = count_tokens_for_conversation(data, conv_idx, include_speaker=False)
        sum_total += total_tok
        print(f"conversation[{conv_idx}] 전체 토큰 = {total_tok:,}")    

    avg_tok = sum_total / 10
    print(f"10개의 대화 평균 토큰 수 = {avg_tok:,}")