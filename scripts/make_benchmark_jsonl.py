import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_BENCHMARK_DIR = Path.cwd() / "Benchmark"
DEFAULT_OUTPUT_DIR = Path.cwd() / "output_jsonl"


def resolve_image_path(benchmark_dir: Path, split: str, relative_image: str) -> Path:
    parts = [part for part in relative_image.replace("\\", "/").split("/") if part]
    return (benchmark_dir / split / Path(*parts)).resolve()


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_singleturn_records(
    benchmark_dir: Path,
    check_images: bool,
) -> List[Dict[str, Any]]:
    annotation_path = benchmark_dir / "singleturn" / "singleturn.json"
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    for index in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        item = data[index]
        image_rel = item["id"]
        image_path = resolve_image_path(benchmark_dir, "singleturn", image_rel)
        if check_images and not image_path.exists():
            missing_images.append(str(image_path))

        records.append(
            {
                "index": str(index),
                "input_image": str(image_path),
                "instruction": item["prompt"],
            }
        )

    if missing_images:
        preview = "\n".join(missing_images[:10])
        print(f"WARNING: Missing singleturn images ({len(missing_images)}):\n{preview}")

    return records


def build_hard_records(
    benchmark_dir: Path,
    check_images: bool,
) -> List[Dict[str, Any]]:
    annotation_path = benchmark_dir / "hard" / "annotation.jsonl"
    records: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    with annotation_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            image_name = item["id"]
            image_path = resolve_image_path(benchmark_dir, "hard", image_name)
            if check_images and not image_path.exists():
                missing_images.append(str(image_path))

            records.append(
                {
                    "index": Path(image_name).stem,
                    "input_image": str(image_path),
                    "instruction": item["prompt"],
                }
            )

    if missing_images:
        preview = "\n".join(missing_images[:10])
        print(f"WARNING: Missing hard images ({len(missing_images)}):\n{preview}")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local ImgEdit Benchmark singleturn/hard annotations into standard JSONL files."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Path to the local Benchmark directory extracted from Benchmark.tar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated singleturn.jsonl and hard.jsonl files.",
    )
    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="Skip checking whether local image files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    check_images = not args.no_check_images

    singleturn_records = build_singleturn_records(benchmark_dir, check_images)
    hard_records = build_hard_records(benchmark_dir, check_images)

    singleturn_out = args.output_dir / "singleturn.jsonl"
    hard_out = args.output_dir / "hard.jsonl"

    singleturn_count = write_jsonl(singleturn_records, singleturn_out)
    hard_count = write_jsonl(hard_records, hard_out)

    print(f"singleturn: {singleturn_count} -> {singleturn_out}")
    print(f"hard: {hard_count} -> {hard_out}")


if __name__ == "__main__":
    main()
