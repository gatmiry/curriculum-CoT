# given a model trained in finetune_generative.py, load in the eval set and test the model using vllm
from typing import Any
import argparse
import os
from datasets import Dataset
from vllm import LLM, SamplingParams
import torch
from statistics import mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="propercache/cache/generative_training/Qwen_Qwen3-Embedding-0.6B_mathtask1_expr2num_min0_max5000_md0_0_hardneg0.0_trainingsize100000")
    parser.add_argument("--eval_set_path", type=str, default="propercache/data/evalsets/mathtask1_expr2num_min0_max5000_md0_0_evalsize500")
    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.eval_set_path)
    # question 

    template = "Input:\n{}\nOutput:\n"
    questions = dataset["question"]
    prompts = [template.format(q) for q in questions]

    breakpoint()
    model = LLM(args.model_path, tensor_parallel_size=torch.cuda.device_count())
    responses = model.generate(prompts, sampling_params=SamplingParams(temperature=0.0))
    responses = [r.outputs[0].text for r in responses]
    answers = [row['pos_chunks'][0] for row in dataset]

    print("Acc: ", mean([r == a for r, a in zip(responses, answers)]))
    breakpoint()
    print("Incorrect: ", list[tuple[Any, str, Any]]([(q, r, a) for q, r, a in zip(questions, responses, answers) if r != a]))
    # print(responses)
