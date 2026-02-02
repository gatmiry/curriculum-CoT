#!/bin/bash

# Parameter sweep for train_parity_2.py over 20 seeds
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

echo "Starting parameter sweep with ${#SEEDS[@]} seeds"
echo "Output directory: $OUTPUTS_DIR"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    # Create unique output folder with datetime and seed
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_DIR="$OUTPUTS_DIR/${TIMESTAMP}_seed_${SEED}"
    mkdir -p "$RUN_DIR"
    
    echo ""
    echo "=============================================="
    echo "Running seed $SEED at $TIMESTAMP"
    echo "Output folder: $RUN_DIR"
    echo "=============================================="
    
    # Run training with outputs directed to the run folder
    cd "$UNSUPERVISED_DIR"
    python train_parity_2.py \
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
        2>&1 | tee "$RUN_DIR/training_log.txt"
    
    # Move the model file to the run directory
    if [ -f "parity_model.pt" ]; then
        mv parity_model.pt "$RUN_DIR/parity_model.pt"
    fi
    
    # Create a run summary file
    echo "{" > "$RUN_DIR/run_info.json"
    echo "  \"seed\": $SEED," >> "$RUN_DIR/run_info.json"
    echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$RUN_DIR/run_info.json"
    echo "  \"parameters\": {" >> "$RUN_DIR/run_info.json"
    echo "    \"n_bits\": $N_BITS," >> "$RUN_DIR/run_info.json"
    echo "    \"k_phases\": $K_PHASES," >> "$RUN_DIR/run_info.json"
    echo "    \"n_layers\": $N_LAYERS," >> "$RUN_DIR/run_info.json"
    echo "    \"n_heads\": $N_HEADS," >> "$RUN_DIR/run_info.json"
    echo "    \"n_embd\": $N_EMBD," >> "$RUN_DIR/run_info.json"
    echo "    \"batch_size\": $BATCH_SIZE," >> "$RUN_DIR/run_info.json"
    echo "    \"iterations_per_phase\": $ITERATIONS_PER_PHASE," >> "$RUN_DIR/run_info.json"
    echo "    \"lr\": \"$LR\"," >> "$RUN_DIR/run_info.json"
    echo "    \"eval_interval\": $EVAL_INTERVAL," >> "$RUN_DIR/run_info.json"
    echo "    \"target_loss\": $TARGET_LOSS," >> "$RUN_DIR/run_info.json"
    echo "    \"multilevel\": false," >> "$RUN_DIR/run_info.json"
    echo "    \"separate_heads\": true," >> "$RUN_DIR/run_info.json"
    echo "    \"truncate_backprop\": false," >> "$RUN_DIR/run_info.json"
    echo "    \"backprop_steps\": 1," >> "$RUN_DIR/run_info.json"
    echo "    \"random_subset\": true," >> "$RUN_DIR/run_info.json"
    echo "    \"remember_rate\": $REMEMBER_RATE" >> "$RUN_DIR/run_info.json"
    echo "  }" >> "$RUN_DIR/run_info.json"
    echo "}" >> "$RUN_DIR/run_info.json"
    
    echo "Completed seed $SEED"
    
    # Small delay to ensure unique timestamps
    sleep 2
done

echo ""
echo "=============================================="
echo "Parameter sweep completed!"
echo "Results saved in: $OUTPUTS_DIR"
echo "=============================================="

