import json
import random

INPUT_PATH  = "locomo10_rag.json"   # 원본 파일
OUTPUT_PATH = "locomo10_qa_no_cat5.json"    # 결과 파일
SAMPLE_SIZE = 50                    # conversation당 추출할 QA 개수

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    data = load_json(INPUT_PATH)
    new_data = {}

    for conv_id, conv in data.items():
        # category==5 제외
        eligible_qas = [qa for qa in conv.get("question", []) if qa.get("category") != 5]

        # 무작위로 최대 SAMPLE_SIZE개 선택
        sampled_qas = random.sample(eligible_qas, k=min(SAMPLE_SIZE, len(eligible_qas)))

        # conversation 본문은 그대로 유지
        new_data[conv_id] = {
            "question": sampled_qas,
            "conversation": conv.get("conversation", {})
        }

    save_json(new_data, OUTPUT_PATH)
    print(f"✅ 완료! '{OUTPUT_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
