# notebook that will take in a retrieval dataset, and finetunes it with a generative model (only using positive)
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict
import torch
from trl import SFTConfig, SFTTrainer

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="propercache/data/colbert_training/mathtask1_expr2num_min0_max5000_md0_0_hardneg0.0_trainingsize100000")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--chat_template", type=str, default="Input:\n{}\nOutput:\n")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16, 
        full_finetuning = True
    )

    dataset = DatasetDict.load_from_disk(args.dataset_path)
    # no lora to keep things comparable

    EOS_TOKEN = tokenizer.eos_token
    def convert_to_text(row): 
        row["text"] = args.chat_template.format(row["query"]) + row["positive"] + EOS_TOKEN
        return row
    dataset = dataset.map(convert_to_text, num_proc=8)

    # dataset['train'] = dataset['train'].select(range(1000))
    print("SANITY CHECK: ", dataset['train'][0]['text'])
    

    outkey = args.model_name.replace('/', '_') + "_" + args.dataset_path.split('/')[-1]
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = SFTConfig(
            per_device_train_batch_size = 64,
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps = 5,
            learning_rate = 2e-5,
            logging_steps = 1,
            lr_scheduler_type = "linear",
            seed = 42,
            save_strategy = "no",
            output_dir = f"propercache/cache/generative_training/{outkey}",
            report_to = "wandb", # Use TrackIO/WandB etc
            eval_strategy = "epoch",
            num_train_epochs = 1,
        ),
    )

    trainer = train_on_responses_only(
        trainer, 
        instruction_part="Input:\n", 
        response_part="Output:\n",
    )

    trainerstats = trainer.train()
    model.save_pretrained(f"propercache/cache/generative_training/{outkey}")
    tokenizer.save_pretrained(f"propercache/cache/generative_training/{outkey}")
    


    
