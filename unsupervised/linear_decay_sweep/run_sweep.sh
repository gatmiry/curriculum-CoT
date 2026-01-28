#!/bin/bash
# Sweep script for train_parity_linear_decay.py
# Varies detect_threshold parameter

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/sweep_configs.json"
TRAIN_SCRIPT="${SCRIPT_DIR}/../train_parity_linear_decay.py"
WORKING_DIR="${SCRIPT_DIR}/.."
OUTPUT_BASE="${SCRIPT_DIR}/outputs"

# Maximum number of parallel jobs (default: 4 since all on one GPU)
MAX_PARALLEL=${1:-4}

echo "=========================================="
echo "Running train_parity_linear_decay.py sweep"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Working directory: ${WORKING_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at ${CONFIG_FILE}"
    echo "Run generate_configs.py first."
    exit 1
fi

# Check if train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

# Detect GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Function to get the GPU with the most free memory
get_best_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        sort -t',' -k2 -nr | head -1 | cut -d',' -f1
}

# Select the best GPU once at the start
BEST_GPU=$(get_best_gpu)
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${BEST_GPU})
echo "Selected GPU ${BEST_GPU} with ${FREE_MEM} MiB free memory"
echo "All jobs will run on GPU ${BEST_GPU}"

# Export CUDA_VISIBLE_DEVICES for all child processes
export CUDA_VISIBLE_DEVICES=${BEST_GPU}

# Get number of configurations
NUM_CONFIGS=$(python3 -c "import json; print(len(json.load(open('${CONFIG_FILE}'))))")
echo "Found ${NUM_CONFIGS} configurations"
echo ""
echo "Running with max ${MAX_PARALLEL} parallel jobs on GPU ${BEST_GPU}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Array to store PIDs
declare -a PIDS
declare -a CONFIG_IDS
declare -a OUTPUT_DIRS

# Function to wait for a slot
wait_for_slot() {
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
                wait ${PIDS[$i]} 2>/dev/null || true
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "✓ Config ${CONFIG_IDS[$i]} (PID ${PIDS[$i]}) completed successfully"
                else
                    echo "✗ Config ${CONFIG_IDS[$i]} (PID ${PIDS[$i]}) failed with exit code ${exit_code}"
                fi
                unset PIDS[$i]
                unset CONFIG_IDS[$i]
                unset OUTPUT_DIRS[$i]
                # Reindex arrays
                PIDS=("${PIDS[@]}")
                CONFIG_IDS=("${CONFIG_IDS[@]}")
                OUTPUT_DIRS=("${OUTPUT_DIRS[@]}")
                return
            fi
        done
        sleep 2
    done
}

# Read and process each configuration
for CONFIG_IDX in $(seq 0 $((NUM_CONFIGS - 1))); do
    # Wait for a slot if needed
    wait_for_slot
    
    # Extract configuration using Python
    CONFIG=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    configs = json.load(f)
config = configs[${CONFIG_IDX}]

# Print values for bash
print(f\"ID={config['id']}\")
print(f\"DETECT_THRESHOLD={config['detect_threshold']}\")
print(f\"N_BITS={config['n_bits']}\")
print(f\"K_PHASES={config['k_phases']}\")
print(f\"N_LAYERS={config['n_layers']}\")
print(f\"N_HEADS={config['n_heads']}\")
print(f\"N_EMBD={config['n_embd']}\")
print(f\"BATCH_SIZE={config['batch_size']}\")
print(f\"ITERATIONS={config['iterations_per_phase']}\")
print(f\"LR={config['lr']}\")
print(f\"EVAL_INTERVAL={config['eval_interval']}\")
print(f\"TARGET_LOSS={config['target_loss']}\")
print(f\"MULTILEVEL={str(config['multilevel']).lower()}\")
print(f\"SEPARATE_HEADS={str(config['separate_heads']).lower()}\")
print(f\"TRUNCATE_BACKPROP={str(config['truncate_backprop']).lower()}\")
print(f\"BACKPROP_STEPS={config['backprop_steps']}\")
print(f\"RANDOM_SUBSET={str(config['random_subset']).lower()}\")
print(f\"SEED={config['seed']}\")
print(f\"REMEMBER_RATE={config['remember_rate']}\")
")
    
    # Parse the config
    eval "$CONFIG"
    
    # Create output directory name (e.g., detect_0.025, detect_0.05, etc.)
    OUTPUT_DIR="${OUTPUT_BASE}/detect_${DETECT_THRESHOLD}"
    mkdir -p "${OUTPUT_DIR}"
    
    echo "Starting configuration ${ID}/${NUM_CONFIGS}"
    echo "  detect_threshold: ${DETECT_THRESHOLD}"
    
    # Build command with all arguments
    CMD="cd ${WORKING_DIR} && python ${TRAIN_SCRIPT}"
    CMD="${CMD} --n_bits ${N_BITS}"
    CMD="${CMD} --k_phases ${K_PHASES}"
    CMD="${CMD} --n_layers ${N_LAYERS}"
    CMD="${CMD} --n_heads ${N_HEADS}"
    CMD="${CMD} --n_embd ${N_EMBD}"
    CMD="${CMD} --batch_size ${BATCH_SIZE}"
    CMD="${CMD} --iterations_per_phase ${ITERATIONS}"
    CMD="${CMD} --lr ${LR}"
    CMD="${CMD} --eval_interval ${EVAL_INTERVAL}"
    CMD="${CMD} --target_loss ${TARGET_LOSS}"
    CMD="${CMD} --backprop_steps ${BACKPROP_STEPS}"
    CMD="${CMD} --seed ${SEED}"
    CMD="${CMD} --remember_rate ${REMEMBER_RATE}"
    CMD="${CMD} --detect_threshold ${DETECT_THRESHOLD}"
    CMD="${CMD} --plots_dir ${OUTPUT_DIR}/plots"
    CMD="${CMD} --plot_data_dir ${OUTPUT_DIR}/plot_data"
    
    # Add boolean flags
    if [ "$MULTILEVEL" = "true" ]; then
        CMD="${CMD} --multilevel"
    fi
    if [ "$SEPARATE_HEADS" = "true" ]; then
        CMD="${CMD} --separate_heads"
    fi
    if [ "$TRUNCATE_BACKPROP" = "true" ]; then
        CMD="${CMD} --truncate_backprop"
    fi
    if [ "$RANDOM_SUBSET" = "true" ]; then
        CMD="${CMD} --random_subset"
    fi
    
    # Run in background, redirect output
    LOG_FILE="${OUTPUT_DIR}/run.log"
    eval "${CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    
    echo "  Started with PID ${PID}"
    
    # Save config to output directory
    python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    configs = json.load(f)
config = configs[${CONFIG_IDX}]
with open('${OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
    
    # Track PID
    PIDS+=($PID)
    CONFIG_IDS+=($ID)
    OUTPUT_DIRS+=("${OUTPUT_DIR}")
done

echo ""
echo "=========================================="
echo "All ${NUM_CONFIGS} configurations started"
echo "=========================================="
echo ""
echo "Waiting for all jobs to complete..."
echo "Logs are in: ${OUTPUT_BASE}/detect_*/run.log"
echo ""

# Wait for remaining jobs
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} 2>/dev/null || true
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Config ${CONFIG_IDS[$i]} (PID ${PIDS[$i]}) completed successfully"
    else
        echo "✗ Config ${CONFIG_IDS[$i]} (PID ${PIDS[$i]}) failed with exit code ${exit_code}"
    fi
done

echo ""
echo "=========================================="
echo "Sweep complete!"
echo "=========================================="
echo "Results saved in: ${OUTPUT_BASE}"

