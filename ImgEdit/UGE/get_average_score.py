import json
import argparse
import re

def extract_scores_and_average(entry: str) -> float:
    # Handle None or non-string cases
    if entry is None or not isinstance(entry, str):
        return None
    
    score_pattern = r'(?:Final\s+)?(?:\*\*)?Score(?:\*\*)?:+\s*(\d+)(?:/\d+)?(?:\s*\([^)]*\))?(?:\*\*)?'
    
    matches = re.findall(score_pattern, entry, re.IGNORECASE)
    
    scores = []
    for match in matches:
        try:
            score = int(match)
            if 1 <= score <= 5:
                scores.append(score)
        except ValueError:
            continue
    
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None

def compute_final_average(result_json_dict):
    total_scores = []
    failed_count = 0
    failed_keys = []
    
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            total_scores.append(avg)
        else:
            failed_count += 1
            failed_keys.append(key)
    
    total = len(result_json_dict)
    success = len(total_scores)
    print(f"\n{'='*50}")
    print(f"UGE Score Calculation Summary:")
    print(f"Total entries: {total}")
    print(f"Valid scores: {success}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Success rate: {success/total*100:.2f}%")
    
    if failed_keys:
        print(f"\nFailed to extract scores from keys:")
        for key in sorted(failed_keys, key=lambda x: int(x)):
            value = result_json_dict[key]
            if isinstance(value, dict):
                print(f"  Key {key}: {value}")
            else:
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  Key {key}: {preview}")
    
    print(f"{'='*50}\n")
    
    if total_scores:
        return round(sum(total_scores) / len(total_scores), 2)
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for all keys and print the final average")
    parser.add_argument('--result_json', type=str, required=True, help='Path to the result JSON file')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_average = compute_final_average(data)

    if final_average is not None:
        print(f"Final Average Score: {final_average}")
        print(f"Result: {final_average}/5.0")
    else:
        print("No valid scores found.")

if __name__ == '__main__':
    main()
