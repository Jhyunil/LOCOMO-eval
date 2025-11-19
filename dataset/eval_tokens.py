# -*- coding: utf-8 -*-
"""
locomo_token_counter.py
────────────────────────────────────────────────────────────
LoCoMo JSON → 특정 (대화 idx, 세션 키) 토큰 수 계산 유틸
pip install tiktoken
"""
from pathlib import Path
import json, tiktoken
from typing import List, Dict, Any, Tuple

ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4o/3.5 계열

# ──────────────────────────────────────────────────────────
# 1) 데이터 로드
# ──────────────────────────────────────────────────────────
def load_json(path: str | Path) -> List[Dict[str, Any]]:
    """LoCoMo 원본 JSON 파일 → 파싱된 리스트 반환"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────
# 2) 세션 텍스트 추출 & 토큰 계산
# ──────────────────────────────────────────────────────────
def _join_text(session_turns: List[Dict[str, Any]]) -> str:
    """'text' 필드만 줄바꿈으로 이어붙여 하나의 문자열 생성"""
    return "\n".join(turn["text"] for turn in session_turns if "text" in turn)


def token_count(text: str) -> int:
    """cl100k_base 기준 토큰 수"""
    return len(ENCODING.encode(text))


def count_tokens_for_session(
    data: List[Dict[str, Any]],
    conv_idx: int,
    session_key: str = "session_27",
) -> Tuple[int, str]:
    """
    (대화 index, 세션 키) 지정 → (토큰 수, 미리보기 문자열) 반환
    - conv_idx : data 리스트에서 몇 번째 대화인지(0-based)
    - session_key : 'session_27'처럼 지정
    """
    conv_obj = data[conv_idx]["conversation"]          # {..., 'session_27': [...], ...}
    if session_key not in conv_obj:
        raise KeyError(f"{session_key=} not found in conversation #{conv_idx}")
    session_turns = conv_obj[session_key]
    text_blob = _join_text(session_turns)
    return token_count(text_blob), text_blob[:120] + "..."  # 앞부분 미리보기까지


# ──────────────────────────────────────────────────────────
# 3) CLI 예시
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="LoCoMo 세션 토큰 수 계산기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            사용 예)
              python locomo_token_counter.py /mnt/data/locomo10.json 0 session_27
        """),
    )
    parser.add_argument("json_path", help="LoCoMo 원본 JSON 경로")
    parser.add_argument("conv_idx", type=int, help="대화 인덱스(0-based)")
    parser.add_argument("session_key", help="세션 키(ex. session_27)")
    args = parser.parse_args()

    data = load_json(args.json_path)
    n_tok, preview = count_tokens_for_session(data, args.conv_idx, args.session_key)
    print(f"conversation[{args.conv_idx}] / {args.session_key} → {n_tok:,} tokens")
    print("----- preview -----")
    print(preview)
