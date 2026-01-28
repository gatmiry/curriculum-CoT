#!/bin/bash

# Sweep script for train_multitarget.py
# Runs 100 configurations with varying fourier_num and flipping_bits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/sweep_configs.json"
TRAIN_SCRIPT="${SCRIPT_DIR}/../train_multitarget.py"
WORKING_DIR="${SCRIPT_DIR}/.."
OUTPUT_BASE="${SCRIPT_DIR}/outputs"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo "Running train_multitarget.py sweep"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Working directory: ${WORKING_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    echo "Run 'python generate_configs.py' first to create it."
    exit 1
fi

# Check if train script exists
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Error: Train script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# Detect available GPUs and pick the one with most free memory
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Find GPU with most free memory (will use this single GPU for all jobs)
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${BEST_GPU} | tr -d ' ')

echo "Selected GPU ${BEST_GPU} with ${FREE_MEM} MiB free memory"
echo "All jobs will run on GPU ${BEST_GPU}"
export CUDA_VISIBLE_DEVICES=${BEST_GPU}

# Count configurations
NUM_CONFIGS=$(python3 -c "import json; print(len(json.load(open('${CONFIG_FILE}'))))")
echo "Found ${NUM_CONFIGS} configurations"
echo ""

# Parse arguments
MAX_PARALLEL=${1:-4}  # Default to 4 parallel jobs on the single GPU
echo "Running with max ${MAX_PARALLEL} parallel jobs on GPU ${BEST_GPU}"
echo ""

# Arrays to track processes
declare -a PIDS
declare -a CONFIG_IDS

# Function to wait for a slot
wait_for_slot() {
    while [ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]; do
        # Check each PID and remove completed ones
        local new_pids=()
        local new_ids=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                new_pids+=("${PIDS[$i]}")
                new_ids+=("${CONFIG_IDS[$i]}")
            else
                wait "${PIDS[$i]}" 2>/dev/null || true
                echo "  ✓ Configuration ${CONFIG_IDS[$i]} completed"
            fi
        done
        PIDS=("${new_pids[@]}")
        CONFIG_IDS=("${new_ids[@]}")
        
        if [ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]; then
            sleep 2
        fi
    done
}

# Run all configurations
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    # Extract config values using Python
    CONFIG=$(python3 -c "
import json
configs = json.load(open('${CONFIG_FILE}'))
c = configs[$i]
print(f\"CONFIG_ID={c['id']}\")
print(f\"FOURIER_NUM={c['fourier_num']}\")
print(f\"FLIPPING_BITS={c['flipping_bits']}\")
print(f\"FOURIER_FILENAME={c.get('fourier_filename', 'fourier' + str(c['fourier_num']))}\")
print(f\"FOURIER_EXPR={c.get('fourier_expression', '')}\")
print(f\"N_BITS={c['n_bits']}\")
print(f\"K_PHASES={c['k_phases']}\")
print(f\"N_LAYERS={c['n_layers']}\")
print(f\"N_HEADS={c['n_heads']}\")
print(f\"N_EMBD={c['n_embd']}\")
print(f\"BATCH_SIZE={c['batch_size']}\")
print(f\"ITERATIONS={c['iterations_per_phase']}\")
print(f\"LR={c['lr']}\")
print(f\"EVAL_INTERVAL={c['eval_interval']}\")
print(f\"EVAL_BATCH_SIZE={c['eval_batch_size']}\")
print(f\"TARGET_LOSS={c['target_loss']}\")
print(f\"MULTILEVEL={c['multilevel']}\")
print(f\"SEPARATE_HEADS={c['separate_heads']}\")
print(f\"TRUNCATE_BACKPROP={c['truncate_backprop']}\")
print(f\"BACKPROP_STEPS={c['backprop_steps']}\")
print(f\"RANDOM_SUBSET={c['random_subset']}\")
print(f\"SEED={c['seed']}\")
print(f\"REMEMBER_RATE={c['remember_rate']}\")
print(f\"DETECT_THRESHOLD={c['detect_threshold']}\")
print(f\"FLIPPING_RATIO={c['flipping_ratio']}\")
")
    eval "${CONFIG}"
    
    # Wait for a slot if needed
    wait_for_slot
    
    # Create output directory for this config
    # Sanitize flipping_bits for folder name (replace comma with dash)
    FLIPPING_BITS_SAFE=$(echo "${FLIPPING_BITS}" | tr ',' '-')
    RUN_DIR="${OUTPUT_BASE}/${FOURIER_FILENAME}_flip${FLIPPING_BITS_SAFE}"
    mkdir -p "${RUN_DIR}/plots"
    mkdir -p "${RUN_DIR}/data"
    mkdir -p "${RUN_DIR}/models"
    
    echo "Starting configuration ${CONFIG_ID}/100"
    echo "  Fourier: ${FOURIER_EXPR}"
    echo "  Flipping bits: ${FLIPPING_BITS}"
    
    # Build command (CUDA_VISIBLE_DEVICES already set globally)
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
    CMD="${CMD} --eval_batch_size ${EVAL_BATCH_SIZE}"
    CMD="${CMD} --target_loss ${TARGET_LOSS}"
    CMD="${CMD} --backprop_steps ${BACKPROP_STEPS}"
    CMD="${CMD} --seed ${SEED}"
    CMD="${CMD} --remember_rate ${REMEMBER_RATE}"
    CMD="${CMD} --detect_threshold ${DETECT_THRESHOLD}"
    CMD="${CMD} --fourier_num ${FOURIER_NUM}"
    CMD="${CMD} --flipping_bits '${FLIPPING_BITS}'"
    CMD="${CMD} --flipping_ratio ${FLIPPING_RATIO}"
    CMD="${CMD} --plots_dir '${RUN_DIR}/plots'"
    CMD="${CMD} --plot_data_dir '${RUN_DIR}/data'"
    
    # Add boolean flags
    if [ "${MULTILEVEL}" = "True" ]; then
        CMD="${CMD} --multilevel"
    fi
    if [ "${SEPARATE_HEADS}" = "True" ]; then
        CMD="${CMD} --separate_heads"
    fi
    if [ "${TRUNCATE_BACKPROP}" = "True" ]; then
        CMD="${CMD} --truncate_backprop"
    fi
    if [ "${RANDOM_SUBSET}" = "True" ]; then
        CMD="${CMD} --random_subset"
    fi
    
    # Run in background with output redirected to log file
    LOG_FILE="${RUN_DIR}/run.log"
    eval "${CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    
    PIDS+=("${PID}")
    CONFIG_IDS+=("${CONFIG_ID}")
    
    echo "  Started with PID ${PID}"
    
    # Save run config to the output directory
    python3 -c "
import json
configs = json.load(open('${CONFIG_FILE}'))
c = configs[$i]
c['assigned_gpu'] = ${BEST_GPU}
c['pid'] = ${PID}
with open('${RUN_DIR}/config.json', 'w') as f:
    json.dump(c, f, indent=2)
"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo ""
echo "=========================================="
echo "All ${NUM_CONFIGS} configurations started"
echo "=========================================="
echo ""
echo "Monitoring processes..."
echo "Logs are being written to: ${OUTPUT_BASE}/*/run.log"
echo ""
echo "To check running processes: ps aux | grep train_multitarget.py"
echo "To check GPU usage: nvidia-smi"
echo "To stop all processes: pkill -f train_multitarget.py"
echo ""
echo "Waiting for all processes to complete..."

# Wait for remaining processes
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Configuration ${CONFIG_IDS[$i]} completed successfully"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Configuration ${CONFIG_IDS[$i]} failed with exit code ${EXIT_CODE}"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "=========================================="
echo "Sweep complete!"
echo "=========================================="
echo "Successful: ${SUCCESS_COUNT}/${NUM_CONFIGS}"
if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Failed: ${FAIL_COUNT}/${NUM_CONFIGS}"
fi
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
