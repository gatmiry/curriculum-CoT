#!/bin/bash

# Parameter sweep for train_parity_2_cot.py over 20 seeds
# Distributed across multiple GPUs for parallelization
# Each run saves outputs to a unique folder with datetime and seed number

# Activate virtual environment
source /mnt/task_runtime/myenv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSUPERVISED_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"

# Configuration parameters (matching config.json)
N_BITS=20
K_PHASES=10
N_LAYERS=1
N_HEADS=1
N_EMBD=64
BATCH_SIZE=128
ITERATIONS_PER_PHASE=15000
LR=1e-5
EVAL_INTERVAL=200
TARGET_LOSS=0.02
REMEMBER_RATE=0.0

# Seeds to sweep over
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

# GPUs to use (avoiding GPU 0 which may be in use)
GPUS=(1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

echo "Starting parameter sweep with ${#SEEDS[@]} seeds across $NUM_GPUS GPUs"
echo "Output directory: $OUTPUTS_DIR"
echo "GPUs: ${GPUS[*]}"
echo "=============================================="

# Function to run a single seed on a specific GPU
run_seed() {
    local SEED=$1
    local GPU=$2
    
    # Create unique output folder with datetime and seed
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_DIR="$OUTPUTS_DIR/${TIMESTAMP}_seed_${SEED}"
    mkdir -p "$RUN_DIR"
    
    echo "[GPU $GPU] Starting seed $SEED at $TIMESTAMP -> $RUN_DIR"
    
    # Run training with outputs directed to the run folder
    cd "$UNSUPERVISED_DIR"
    CUDA_VISIBLE_DEVICES=$GPU python train_parity_2_cot.py \
        --n_bits $N_BITS \
        --k_phases $K_PHASES \
        --n_layers $N_LAYERS \
        --n_heads $N_HEADS \
        --n_embd $N_EMBD \
        --batch_size $BATCH_SIZE \
        --iterations_per_phase $ITERATIONS_PER_PHASE \
        --lr $LR \
        --eval_interval $EVAL_INTERVAL \
        --target_loss $TARGET_LOSS \
        --remember_rate $REMEMBER_RATE \
        --seed $SEED \
        --plots_dir "$RUN_DIR" \
        --plot_data_dir "$RUN_DIR" \
        > "$RUN_DIR/training_log.txt" 2>&1
    
    # Move the model file to the run directory
    if [ -f "$UNSUPERVISED_DIR/parity_model.pt" ]; then
        mv "$UNSUPERVISED_DIR/parity_model.pt" "$RUN_DIR/parity_model.pt" 2>/dev/null || true
    fi
    
    # Create a run summary file
    cat > "$RUN_DIR/run_info.json" << EOF
{
  "seed": $SEED,
  "gpu": $GPU,
  "timestamp": "$TIMESTAMP",
  "parameters": {
    "n_bits": $N_BITS,
    "k_phases": $K_PHASES,
    "n_layers": $N_LAYERS,
    "n_heads": $N_HEADS,
    "n_embd": $N_EMBD,
    "batch_size": $BATCH_SIZE,
    "iterations_per_phase": $ITERATIONS_PER_PHASE,
    "lr": "$LR",
    "eval_interval": $EVAL_INTERVAL,
    "target_loss": $TARGET_LOSS,
    "multilevel": false,
    "separate_heads": true,
    "truncate_backprop": false,
    "backprop_steps": 1,
    "random_subset": true,
    "remember_rate": $REMEMBER_RATE
  }
}
EOF
    
    echo "[GPU $GPU] Completed seed $SEED"
}

# Export function and variables for parallel execution
export -f run_seed
export OUTPUTS_DIR UNSUPERVISED_DIR
export N_BITS K_PHASES N_LAYERS N_HEADS N_EMBD BATCH_SIZE
export ITERATIONS_PER_PHASE LR EVAL_INTERVAL TARGET_LOSS REMEMBER_RATE

# Run seeds in parallel across GPUs
# Process seeds in batches, with each GPU handling multiple seeds sequentially
for ((i=0; i<${#SEEDS[@]}; i+=NUM_GPUS)); do
    PIDS=()
    
    for ((j=0; j<NUM_GPUS && i+j<${#SEEDS[@]}; j++)); do
        SEED=${SEEDS[$((i+j))]}
        GPU=${GPUS[$j]}
        
        run_seed $SEED $GPU &
        PIDS+=($!)
    done
    
    # Wait for this batch to complete
    echo "Waiting for batch starting at seed $((i)) to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    echo "Batch complete."
done

echo ""
echo "=============================================="
echo "Parameter sweep completed!"
echo "Results saved in: $OUTPUTS_DIR"
echo "=============================================="

