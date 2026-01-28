#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4
# Determine project glue directory relative to this script if not provided
GLUE_DIR=${GLUE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/glue"}
cd "$GLUE_DIR"

MODEL_PATH="output/gpt2-large_lora_e2e_sa"
BASE_MODEL="gpt2-large"
TEST_FILE="data/e2e_nlg/testset_w_refs.csv"

echo "=========================================="
echo "E2E Official e2e-metrics Evaluation"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

if [ ! -d "e2e-metrics" ]; then
    echo ""
    cd e2e-metrics
    pip install -r requirements.txt
    pip install future
    cd ..
else
    echo "e2e-metrics already exists, skipping clone"
fi

echo ""
echo "Generating prediction text..."
python src/eval_e2e_generate.py \
    --model_path $MODEL_PATH \
    --base_model_name $BASE_MODEL \
    --test_file $TEST_FILE \
    --output_file ${MODEL_PATH}/system_output.txt

echo ""
echo "Preparing reference file..."
python - <<EOF
import csv
from collections import OrderedDict

test_file = "$TEST_FILE"
output_ref = "${MODEL_PATH}/reference_correct.txt"

mr_to_refs = OrderedDict()
with open(test_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        mr = row['mr']
        ref = row['ref']
        if mr not in mr_to_refs:
            mr_to_refs[mr] = []
        mr_to_refs[mr].append(ref)

with open(output_ref, 'w', encoding='utf-8') as f:
    first = True
    for mr, refs in mr_to_refs.items():
        if not first:
            f.write('\n')
        first = False
        for ref in refs:
            f.write(ref + '\n')

print(f"Reference file saved: {output_ref}")
print(f"Total MRs: {len(mr_to_refs)}")
EOF

echo ""
echo "Running official e2e-metrics evaluation..."
echo "=========================================="
cd e2e-metrics
python measure_scores.py \
    ../${MODEL_PATH}/reference_correct.txt \
    ../${MODEL_PATH}/system_output.txt
cd ..

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
