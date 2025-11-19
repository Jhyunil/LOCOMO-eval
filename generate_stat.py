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

# eval_data 와 qa_data 는 각 세션별로 QA 리스트가 들어 있는 dict 형태
# 각 세션의 QA 리스트를 풀어서 하나의 list 로 만든 후 DataFrame 으로 변환

eval_items = [it for conv in eval_data.values() for it in conv]
qa_items   = [it for conv in qa_data.values()  for it in conv]

# DataFrame 생성

df_eval = pd.DataFrame(eval_items)
df_qa   = pd.DataFrame(qa_items)[["question", "response_time", "total_tokens", "completion_reasoning_tokens"]]

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

# ---------- 5) 카테고리별 집계 (평균) ----------
group_cols = [
    "bleu_score",
    "f1_score",
    "llm_score",
    "response_time",
    "total_tokens",
]

result = (
    df.groupby("category")[group_cols]
      .mean()
      .round(4)
      .assign(count=df.groupby("category").size())
)

# ---------- 5‑b) 카테고리별 response_time 퍼센타일(p50, p95) 추가 ----------
#   • p50: 중앙값(median)
#   • p95: 95th percentile

percentiles = (
    df.groupby("category")["response_time"]
      .quantile([0.5, 0.95])
      .unstack()
      .rename(columns={0.5: "p50_response_time", 0.95: "p95_response_time"})
      .round(4)
)

# 평균 결과 테이블(result) 에 퍼센타일 붙이기
result = result.join(percentiles)

print("Mean Scores + Runtime/Token Usage Per Category (incl. p50/p95):")
print(result)

# ---------- 6) 전체 평균 + 총 토큰 합계 + 전체 퍼센타일 ----------

overall = df[group_cols].mean().round(4)

overall["total_tokens_sum"] = int(df["total_tokens"].sum())
overall["reasoning_tokens_sum"] = int(df["completion_reasoning_tokens"].sum())
# 전체 p50 / p95
overall["response_time_p50"] = round(df["response_time"].median(), 4)
overall["response_time_p95"] = round(df["response_time"].quantile(0.95), 4)
overall["response_min"] = round(df["response_time"].min(), 4)
overall["response_max"] = round(df["response_time"].max(), 4)

# --- NEW: 지수 표기 없애기 ---
pd.set_option("display.float_format", "{:.4f}".format)   # 4자리 고정소수점
print("\nOverall Mean Scores + Runtime/Token Usage (incl. p50/p95):")
print(overall)
pd.reset_option("display.float_format")                  # 옵션 원상 복구


# df["response_time"]을 가장 큰 값 10개
print("\nResponse Time Sorted:")
print(df.sort_values(by="response_time")[["question", "response_time"]].tail(10))