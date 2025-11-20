import json

def calculate_decode_time_avg(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    decode_times = []
    prefill_times = []

    # data: {"0": [...], "1": [...], ...}
    for key, item_list in data.items():
        if not isinstance(item_list, list):
            continue
        for item in item_list:
            if "decode_time_avg" in item:
                decode_times.append(item["decode_time_avg"])
            if "prefill_time" in item:
                prefill_times.append(item["prefill_time"])

    if len(decode_times) == 0 or len(prefill_times) == 0:
        print("No decode_time_avg fields found.")
        return

    decode_avg = sum(decode_times) / len(decode_times)
    prefill_avg = sum(prefill_times) / len(prefill_times)
    print(f"Total items: {len(decode_times)}")

    print(f"Average prefill_time_avg: {prefill_avg:.6f}")
    print(f"Average decode_time_avg: {decode_avg:.6f}")

if __name__ == "__main__":
    print('4conv input token with vllm: ')
    calculate_decode_time_avg("results/gpt-oss/gpt-oss_4conv_vllm.json")
    print()

    print('1conv input token with vllm: ')
    calculate_decode_time_avg("results/gpt-oss/gpt-oss_1conv_vllm.json")
    print()

    print('1conv input token with vllm with prefix cache: ')
    calculate_decode_time_avg("results/gpt-oss/gpt-oss_1conv_prefix_vllm.json")
    print()
