import json

def calculate_decode_time_avg(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    decode_times = []

    # data: {"0": [...], "1": [...], ...}
    for key, item_list in data.items():
        if not isinstance(item_list, list):
            continue
        for item in item_list:
            if "decode_time_avg" in item:
                decode_times.append(item["decode_time_avg"])

    if len(decode_times) == 0:
        print("No decode_time_avg fields found.")
        return

    avg = sum(decode_times) / len(decode_times)
    print(f"Total items: {len(decode_times)}")
    print(f"Average decode_time_avg: {avg:.6f}")

if __name__ == "__main__":
    calculate_decode_time_avg("results/gpt-oss/gpt-oss_8192trt_remote.json")
