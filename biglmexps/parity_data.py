# file to generate string parity task which can be finetuned on
import argparse
import random
from datasets import Dataset, DatasetDict
import os

def generate_parity_example(n_bits, min_positions=2, max_positions=None):
    """
    Generate a single parity example.
    
    Args:
        n_bits: Total number of bits
        min_positions: Minimum number of positions to select
        max_positions: Maximum number of positions to select (default: n_bits)
    
    Returns:
        tuple: (query_string, answer_string)
    """
    if max_positions is None:
        max_positions = n_bits
    
    # Select random number of positions
    num_positions = random.randint(min_positions, min(max_positions, n_bits))
    
    # Select random positions (without replacement)
    positions = sorted(random.sample(range(n_bits), num_positions))
    
    # Generate random bit values (-1 or 1)
    bit_values = [random.choice([-1, 1]) for _ in range(n_bits)]
    
    # Format positions as "0x3x5" style
    position_str = "x".join(str(p) for p in positions)
    
    # Format bit values as space-separated string
    bit_values_str = " ".join(str(b) for b in bit_values)
    
    # Create query: "0x3x5 1 1 1 1 -1 -1 -1"
    query = f"{position_str} {bit_values_str}"
    
    # Compute answer: product of values at selected positions
    answer = 1
    for pos in positions:
        answer *= bit_values[pos]
    
    answer_str = str(answer)
    
    return query, answer_str

def generate_dataset(n_bits, num_train, num_eval, min_positions=2, max_positions=None, seed=42):
    """
    Generate training and evaluation datasets for the parity task.
    
    Args:
        n_bits: Number of bits
        num_train: Number of training examples
        num_eval: Number of evaluation examples
        min_positions: Minimum number of positions to select
        max_positions: Maximum number of positions to select
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset_dict, eval_dataset)
    """
    random.seed(seed)
    
    # Generate training examples
    train_queries = []
    train_answers = []
    for _ in range(num_train):
        query, answer = generate_parity_example(n_bits, min_positions, max_positions)
        train_queries.append(query)
        train_answers.append(answer)
    
    # Generate eval examples
    eval_queries = []
    eval_answers = []
    for _ in range(num_eval):
        query, answer = generate_parity_example(n_bits, min_positions, max_positions)
        eval_queries.append(query)
        eval_answers.append(answer)
    
    # Create training dataset in format expected by finetune.py
    # finetune.py expects: query, positive
    train_dataset = Dataset.from_dict({
        "query": train_queries,
        "positive": train_answers
    })
    
    # Create test split (also used for eval during training)
    test_dataset = Dataset.from_dict({
        "query": eval_queries,
        "positive": eval_answers
    })
    
    # Create DatasetDict for training
    train_dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Create eval dataset in format expected by test_generative.py
    # test_generative.py expects: question, pos_chunks (list)
    eval_dataset = Dataset.from_dict({
        "question": eval_queries,
        "pos_chunks": [[ans] for ans in eval_answers]  # pos_chunks is a list of lists
    })
    
    return train_dataset_dict, eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parity task dataset")
    parser.add_argument("--n_bits", type=int, required=True, help="Number of bits")
    parser.add_argument("--num_train", type=int, default=100000, help="Number of training examples")
    parser.add_argument("--num_eval", type=int, default=500, help="Number of evaluation examples")
    parser.add_argument("--min_positions", type=int, default=2, help="Minimum number of positions to select")
    parser.add_argument("--max_positions", type=int, default=None, help="Maximum number of positions to select (default: n_bits)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_output_path", type=str, default=None, help="Path to save training dataset")
    parser.add_argument("--eval_output_path", type=str, default=None, help="Path to save eval dataset")
    
    args = parser.parse_args()
    
    # Generate datasets
    train_dataset_dict, eval_dataset = generate_dataset(
        n_bits=args.n_bits,
        num_train=args.num_train,
        num_eval=args.num_eval,
        min_positions=args.min_positions,
        max_positions=args.max_positions,
        seed=args.seed
    )
    
    # Set default output paths if not provided
    if args.train_output_path is None:
        args.train_output_path = f"propercache/data/colbert_training/parity_nbits{args.n_bits}_train{args.num_train}"
    if args.eval_output_path is None:
        args.eval_output_path = f"propercache/data/evalsets/parity_nbits{args.n_bits}_evalsize{args.num_eval}"
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_output_path), exist_ok=True)
    
    # breakpoint()
    # Save datasets
    print(f"Saving training dataset to {args.train_output_path}")
    train_dataset_dict.save_to_disk(args.train_output_path)
    
    print(f"Saving eval dataset to {args.eval_output_path}")
    eval_dataset.save_to_disk(args.eval_output_path)
    
    # Print some examples
    print("\nTraining examples:")
    for i in range(min(3, len(train_dataset_dict["train"]))):
        row = train_dataset_dict["train"][i]
        print(f"  Query: {row['query']}")
        print(f"  Answer: {row['positive']}")
    
    print("\nEval examples:")
    for i in range(min(3, len(eval_dataset))):
        row = eval_dataset[i]
        print(f"  Question: {row['question']}")
        print(f"  Answer: {row['pos_chunks'][0]}")
    
    print(f"\nGenerated {args.num_train} training examples and {args.num_eval} eval examples with {args.n_bits} bits")