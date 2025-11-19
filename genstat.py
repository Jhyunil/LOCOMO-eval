import argparse
import json
import pandas as pd


parser = argparse.ArgumentParser(description="Generate evaluation statistics from results")
parser.add_argument(
    "--eval_data", type=str, default="results/evaluation/full_conv5_1.json", help="Path to the eval dataset file"
)
parser.add_argument(
    "--qa_data", type=str, default="results/full_conv5/qa_result_full_conv5_1.json", help="Path to the qa dataset file"
)
parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")
args = parser.parse_args()

# ---------- 1) 두 파일 로드 ----------
with open(args.eval_data) as f:
    eval_data = json.load(f)

with open(args.qa_data) as f:
    qa_data = json.load(f)

# --------------------------------------------------
# 2) Flatten nested lists → DataFrames
# --------------------------------------------------

# eval_data 와 qa_data 는 각 세션별로 QA 리스트가 들어 있는 dict 형태
# 각 세션의 QA 리스트를 풀어서 하나의 list 로 만든 후 DataFrame 으로 변환

eval_items = [it for conv in eval_data.values() for it in conv]
qa_items   = [it for conv in qa_data.values()  for it in conv]

# DataFrame 생성
qa_keep_cols = [
    "question",
    "response_time",
    "prefill_time",
    "decode_time_avg",
    "total_tokens",
    "prompt_tokens",
    "prompt_cached_tokens",
    "completion_reasoning_tokens",
]

df_eval = pd.DataFrame(eval_items)
df_qa   = pd.DataFrame(qa_items)[qa_keep_cols]

dups = (df_eval["question"]
        .value_counts()
        .loc[lambda x: x > 1])
print(f"중복된 질문 수: {len(dups)}")
print(dups.head())
print()

df_eval["dup_idx"] = df_eval.groupby("question").cumcount()
df_qa["dup_idx"]   = df_qa.groupby("question").cumcount()

# ---------- 2) 머지 ----------
# question 컬럼을 키로 두 DF 합치기 – evaluation 측 정보에 QA 측 메트릭(응답시간·토큰) 붙이기

df = pd.merge(df_eval, df_qa, on=["question", "dup_idx"], how="left", suffixes=("", "_qa"), validate="one_to_one")

# ---------- 3) 중복 컬럼 정리 ----------
# evaluation 에도 동일한 컬럼명이 있을 경우 QA 쪽(_qa) 값을 본 컬럼으로 덮어쓰기

for col in ["response_time", "total_tokens"]:
    qa_col = f"{col}_qa"
    if qa_col in df.columns:
        df[col] = df[qa_col]
        df.drop(columns=[qa_col], inplace=True)

# ---------- 4) 전처리 ----------
num_cols = ["category", "response_time", "total_tokens"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# 5) Category‑level aggregation (legacy)
# --------------------------------------------------

group_cols = [
    "bleu_score",
    "f1_score",
    "llm_score",
    "response_time",
    "total_tokens",
]

result_category = (
    df.groupby("category")[group_cols]
      .mean()
      .round(4)
      .assign(count=df.groupby("category").size())
)

percentiles = (
    df.groupby("category")["response_time"].quantile([0.5, 0.95])
      .unstack()
      .rename(columns={0.5: "p50_response_time", 0.95: "p95_response_time"})
      .round(4)
)

result_category = result_category.join(percentiles)

print("\n=== Mean Scores + Runtime / Token Usage per Category (incl. p50 & p95) ===")
print(result_category)

# --------------------------------------------------
# 6) Prompt‑cache aware statistics
# --------------------------------------------------
print("\n=== Prompt‑cache analysis ===")

# Helper -------------------------------------------------------------------

def describe_timing(sub_df: pd.DataFrame, label: str):
    """Print mean, p50, p95, min, max for timing columns."""
    timing_cols = ["prefill_time", "decode_time_avg", "response_time"]
    stats_lines = []
    for col in timing_cols:
        mean = sub_df[col].mean()
        p50  = sub_df[col].median()
        p95  = sub_df[col].quantile(0.95)
        tmin = sub_df[col].min()
        tmax = sub_df[col].max()
        stats_lines.append(
            f"{col:17s} | mean:{mean:8.4f} | min:{tmin:8.4f} | p50:{p50:8.4f} | p95:{p95:8.4f} | max:{tmax:8.4f}"
        )
    print(f"\n-- {label} --")
    print("\n".join(stats_lines))

# Split data ---------------------------------------------------------------

cached_mask     = df["prompt_cached_tokens"].fillna(0) != 0
non_cached_mask = ~cached_mask

df_cached     = df[cached_mask]
df_non_cached = df[non_cached_mask]

# 1. Counts ----------------------------------------------------------------

print(f"QAs w/  cached prompt tokens (>0): {len(df_cached)}")
print(f"QAs w/o cached prompt tokens ( =0): {len(df_non_cached)}")

# 2. Avg prompt_cached_tokens for cached rows ------------------------------

if not df_cached.empty:
    avg_cached_tokens = df_cached["prompt_cached_tokens"].mean()
    print(f"\nAverage prompt_cached_tokens (cached set): {avg_cached_tokens:.2f}")
else:
    print("\n[Warning] No rows with prompt_cached_tokens > 0 found.")

# 3. Timing stats (mean, p50, p95, min, max) --------------------------------

describe_timing(df_cached,     "Rows with prompt_cached_tokens > 0")
describe_timing(df_non_cached, "Rows with prompt_cached_tokens = 0")
describe_timing(df,            "All rows (overall)")

# --------------------------------------------------
# 7) Overall summary (extended)
# --------------------------------------------------

overall_cols = ["bleu_score", "f1_score", "llm_score", "response_time", "total_tokens"]
overall = df[overall_cols].mean().round(4)

overall["total_tokens_sum"]   = int(df["total_tokens"].sum())
overall["reasoning_tokens_sum"] = int(df["completion_reasoning_tokens"].sum())
overall["response_time_p50"] = round(df["response_time"].median(), 4)
overall["response_time_p95"] = round(df["response_time"].quantile(0.95), 4)
overall["response_min"]       = round(df["response_time"].min(), 4)
overall["response_max"]       = round(df["response_time"].max(), 4)

pd.options.display.float_format = '{:.4f}'.format
print("\n=== Overall Mean Scores + Runtime / Token Usage (incl. p50, p95, min, max) ===")
print(overall)

# --------------------------------------------------
# 8) Slowest responses (top‑10) for manual inspection
# --------------------------------------------------

print("\n=== Top‑10 largest response_time ===")
print(
    df.sort_values(by="response_time", ascending=False)[
        ["question", "response_time", "prefill_time", "decode_time_avg", "prompt_cached_tokens"]
    ].head(10)
)
