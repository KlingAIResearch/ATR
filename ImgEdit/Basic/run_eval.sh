#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Error: Missing target folder path!"
    echo "Usage: $0 <result_folder_path>"
    exit 1
fi

TARGET_DIR="${1%/}"

WORK_DIR="./examples/ImgEdit/Benchmark/Basic"
ORIGIN_IMG_ROOT="./examples/ImgEdit/Benchmark/imgeasy_original_images"

cd "$WORK_DIR"

echo "=================================================="
echo "Step 1: Running basic_bench.py"
echo "=================================================="
python basic_bench.py \
  --result_img_folder "$TARGET_DIR" \
  --edit_json ./imgeasy_edit.json \
  --origin_img_root "$ORIGIN_IMG_ROOT" \
  --prompts_json ./prompts.json \
  --num_processes 8

echo "=================================================="
echo "Step 2: Running step1_get_avgscore.py"
echo "=================================================="
python step1_get_avgscore.py \
  --result_json "$TARGET_DIR/result.json" \
  --average_score_json "$TARGET_DIR/average_scores.json"

echo "=================================================="
echo "Step 3: Running step2_typescore.py"
echo "=================================================="
python step2_typescore.py \
  --average_score_json "$TARGET_DIR/average_scores.json" \
  --basic_edit ./basic_edit.json \
  --typescore_json "$TARGET_DIR/type_scores.json"

echo "=================================================="
echo "All evaluation steps completed!"
echo "Results saved to: $TARGET_DIR"
echo "   - $TARGET_DIR/result.json"
echo "   - $TARGET_DIR/average_scores.json"
echo "   - $TARGET_DIR/type_scores.json"
