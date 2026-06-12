import json
import argparse

def extract_scores_and_average(entry: str) -> float:
    # Handle None or non-string cases
    if entry is None or not isinstance(entry, str):
        return None
    
    lines = entry.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(': ')
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None

def compute_averages(result_json_dict):
    result = {}
    failed_count = 0
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
        else:
            failed_count += 1
    
    total = len(result_json_dict)
    success = len(result)
    print(f"\n{'='*50}")
    print(f"Score Extraction Summary:")
    print(f"Total entries: {total}")
    print(f"Successfully extracted: {success}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Success rate: {success/total*100:.2f}%")
    print(f"{'='*50}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for each key and save it as a new JSON file")
    parser.add_argument('--result_json', type=str, required=True, help='Path to the result JSON file')
    parser.add_argument('--average_score_json', type=str, required=True, help='Path to the output average score JSON file')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    averaged_data = compute_averages(data)

    with open(args.average_score_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_data, f, indent=2)


if __name__ == '__main__':
    main()
