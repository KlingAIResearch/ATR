#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Error: Missing target folder path!"
    echo "Usage: $0 <result_folder_path>"
    exit 1
fi

TARGET_DIR="${1%/}"

# Define fixed base paths and script paths
BASE_DIR="./examples/ImgEdit/Benchmark/UGE"
UGE_BENCH_SCRIPT="$BASE_DIR/UGE_bench.py"
GET_SCORE_SCRIPT="$BASE_DIR/get_average_score.py"
EDIT_JSON="$BASE_DIR/UGE_edit.json"
ORIGIN_IMG_ROOT="$BASE_DIR/uge_original_images"

echo "=================================================="
echo "Step 1: Running UGE_bench.py"
echo "=================================================="
python "$UGE_BENCH_SCRIPT" \
  --result_img_folder "$TARGET_DIR" \
  --edit_json "$EDIT_JSON" \
  --origin_img_root "$ORIGIN_IMG_ROOT" \
  --num_processes 4

echo "--------------------------------------------------"
echo "Step 2: Running get_average_score.py"
python "$GET_SCORE_SCRIPT" \
  --result_json "$TARGET_DIR/result.json"

echo "=================================================="
echo "UGE evaluation completed!"
