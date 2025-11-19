import json

# ===== 1) 파일 읽기 =====
# 파일 경로를 작업 환경에 맞게 바꿔 주세요.
FNAME = "qa_result_including_cat5.json"

with open(FNAME, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===== 2) 모든 항목(flatten) =====
# data는 key가 대화 ID(예: "0", "1", …)이고 값이 질문-응답 리스트인 구조
all_items = []
for conv_items in data.values():
    all_items.extend(conv_items)

# ===== 3) total_tokens 합계 계산 =====
total = sum(item.get("total_tokens", 0) for item in all_items)

# ===== 4) 결과 출력 =====
print(f"총 token 사용량: {total:,}")
