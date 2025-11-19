import json
import random

INPUT_PATH  = "locomo10_rag.json"   # 원본 파일
OUTPUT_PATH = "locomo10_qa.json"    # 결과 파일

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    for conv_id, conv in data.items():
        qa_list = conv.get("question", [])
        # 무작위 50개(질문-답변 쌍) 추출; 50개 미만이면 전부 사용
        sampled_qas = random.sample(qa_list, k=min(50, len(qa_list)))

        # conversation 본문은 그대로 유지
        new_data[conv_id] = {
            "question": sampled_qas,
            "conversation": conv.get("conversation", {})
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"완료! {OUTPUT_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    main()

