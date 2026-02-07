#!/bin/bash
# Run comparison sweep for both CoT and No CoT models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source /mnt/task_runtime/myenv/bin/activate

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
COT_OUTPUT_DIR="comparison_sweep_1/cot_outputs"
NOCOT_OUTPUT_DIR="comparison_sweep_1/nocot_outputs"

# GPUs for CoT: 0-3, for No CoT: 4-7
COT_GPUS=(0 1 2 3)
NOCOT_GPUS=(4 5 6 7)

echo "Starting comparison sweep at $(date)"
echo "Running CoT sweep on GPUs: ${COT_GPUS[*]}"
echo "Running No CoT sweep on GPUs: ${NOCOT_GPUS[*]}"

# Run CoT sweep
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU_IDX=$((i % 4))
    GPU=${COT_GPUS[$GPU_IDX]}
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${COT_OUTPUT_DIR}/${TIMESTAMP}_seed_${SEED}"
    mkdir -p "$RUN_DIR"
    
    echo "[CoT] Starting seed $SEED on GPU $GPU -> $RUN_DIR"
    
    CUDA_VISIBLE_DEVICES=$GPU python train_parity_2_cot.py \
        --seed $SEED \
        --plots_dir "$RUN_DIR/plots" \
        --plot_data_dir "$RUN_DIR/plot_data" \
        > "$RUN_DIR/train.log" 2>&1 &
    
    # Small delay to avoid timestamp collision
    sleep 1
done

# Run No CoT sweep
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU_IDX=$((i % 4))
    GPU=${NOCOT_GPUS[$GPU_IDX]}
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${NOCOT_OUTPUT_DIR}/${TIMESTAMP}_seed_${SEED}"
    mkdir -p "$RUN_DIR"
    
    echo "[No CoT] Starting seed $SEED on GPU $GPU -> $RUN_DIR"
    
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
echo "Sweep completed at $(date)"

# Generate comparison plots
echo "Generating comparison plots..."
python comparison_sweep_1/plot_comparison.py

echo "Done!"
