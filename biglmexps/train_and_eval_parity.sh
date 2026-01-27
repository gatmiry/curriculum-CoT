N_BITS=10
MODEL_NAME="facebook/opt-125m"
NUM_TRAIN=100000
NUM_EVAL=500

# Step 1: Generate data
echo "=========================================="
echo "Step 1: Generating parity data with n_bits=$N_BITS"
echo "=========================================="
python biglmexps/parity_data.py --n_bits $N_BITS --num_train $NUM_TRAIN --num_eval $NUM_EVAL

# Set paths based on generated data
TRAIN_DATASET_PATH="propercache/data/colbert_training/parity_nbits${N_BITS}_train${NUM_TRAIN}"
EVAL_DATASET_PATH="propercache/data/evalsets/parity_nbits${N_BITS}_evalsize${NUM_EVAL}"

# Construct model output path (matching finetune.py logic)
MODEL_NAME_UNDERSCORE=$(echo $MODEL_NAME | tr '/' '_')
DATASET_NAME=$(basename $TRAIN_DATASET_PATH)
MODEL_OUTPUT_PATH="propercache/cache/generative_training/${MODEL_NAME_UNDERSCORE}_${DATASET_NAME}"

# Step 2: Train the model
echo ""
echo "=========================================="
echo "Step 2: Training model on parity data"
echo "=========================================="
echo "Training dataset: $TRAIN_DATASET_PATH"
echo "Model will be saved to: $MODEL_OUTPUT_PATH"
accelerate launch biglmexps/finetune.py \
    --dataset_path "$TRAIN_DATASET_PATH" \
    --model_name "$MODEL_NAME"

# Step 3: Evaluate the model
echo ""
echo "=========================================="
echo "Step 3: Evaluating model on test set"
echo "=========================================="
echo "Model path: $MODEL_OUTPUT_PATH"
echo "Eval dataset: $EVAL_DATASET_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biglmexps/test_generative.py \
    --model_path "$MODEL_OUTPUT_PATH" \
    --eval_set_path "$EVAL_DATASET_PATH"

echo ""
echo "=========================================="
echo "Training and evaluation complete!"
echo "=========================================="
