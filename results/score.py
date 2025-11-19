FPath = 'evaluation/full_conv1.json'
output = 'analysis/llm_score_correct_full_conv1.json'
import json
import pandas as pd

# f1, bleu score가 0.5 미만이면서 llm_score가 1인 데이터를 찾아서 저장
def find_zero_llm_score(data):
    zero_llm_data = []
    for key, items in data.items():
        for item in items:
            if item['f1_score'] < 0.5 and item['bleu_score'] < 0.5 and item['llm_score'] == 1:
                zero_llm_data.append(item)
    return zero_llm_data

# main function에서 위의 함수를 호출하여 결과를 output 파일에 저장
def main():
    with open(FPath, 'r') as f:
        data = json.load(f)

    zero_llm_data = find_zero_llm_score(data)

    # Save the results to a JSON file
    with open(output, 'w') as f:
        json.dump(zero_llm_data, f, indent=4)

    print(f"Found {len(zero_llm_data)} items with f1_score < 0.5, bleu_score < 0.5 and llm_score == 1.")

if __name__ == "__main__":
    main()
