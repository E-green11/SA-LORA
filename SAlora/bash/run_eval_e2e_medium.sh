#!/bin/bash
# E2E official evaluation script (GPT2-Medium)

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4
GLUE_DIR=${GLUE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/glue"}
cd "$GLUE_DIR"

# ========== Configuration ==========
MODEL_PATH="output/gpt2-medium_lora_e2e_sa"
BASE_MODEL="gpt2-medium"
TEST_FILE="data/e2e_nlg/testset_w_refs.csv"

echo "=========================================="
echo "E2E official evaluation (Official e2e-metrics)"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

# Step 1: Clone official e2e-metrics if missing
if [ ! -d "e2e-metrics" ]; then
    echo ""
    echo "[Step 1] Cloning official e2e-metrics..."
    
    cd e2e-metrics
    pip install -r requirements.txt
    pip install future
    cd ..
else
    echo "[Step 1] e2e-metrics already exists, skipping clone"
fi

# Step 2: Generate prediction text
echo ""
echo "[Step 2] Generating prediction text..."
python src/eval_e2e_generate.py \
    --model_path $MODEL_PATH \
    --base_model_name $BASE_MODEL \
    --test_file $TEST_FILE \
    --output_file ${MODEL_PATH}/system_output.txt

# Step 3: Prepare reference file in the correct format
echo ""
echo "[Step 3] Preparing reference file (one ref per line, blank line between MRs)..."
python - <<'EOF'
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

# Step 4: Run official evaluation
echo ""
echo "[Step 4] Running official e2e-metrics evaluation..."
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
