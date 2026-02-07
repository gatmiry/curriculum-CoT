#!/bin/bash
# Run no-curriculum sweep using train_parity_2.py (trains directly on n_bits)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source /mnt/task_runtime/myenv/bin/activate

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
OUTPUT_DIR="comparison_sweep_1/nocurriculum_outputs"

# Use all 8 GPUs
GPUS=(0 1 2 3 4 5 6 7)

echo "Starting no-curriculum sweep at $(date)"
echo "Using GPUs: ${GPUS[*]}"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU_IDX=$((i % 8))
    GPU=${GPUS[$GPU_IDX]}
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_seed_${SEED}"
    mkdir -p "$RUN_DIR"
    
    echo "[No Curriculum] Starting seed $SEED on GPU $GPU -> $RUN_DIR"
    
    CUDA_VISIBLE_DEVICES=$GPU python train_parity_2.py \
        --seed $SEED \
        --plots_dir "$RUN_DIR/plots" \
        --plot_data_dir "$RUN_DIR/plot_data" \
        > "$RUN_DIR/train.log" 2>&1 &
    
    # Small delay to avoid timestamp collision
    sleep 1
done

echo ""
echo "All jobs launched. Waiting for completion..."
wait

echo ""
echo "No-curriculum sweep completed at $(date)"
echo "Done!"
