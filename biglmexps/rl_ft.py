from datasets import DatasetDict
from trl import GRPOTrainer, GRPOConfig
import argparse
from trl.rewards import accuracy_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trl-lib/DeepMath-103K")
    args = parser.parse_args()

    # load in data (sft format, need to convert to rl format)
    dataset = DatasetDict.load_from_disk(args.dataset_name)

    def em_reward(completions=None, solution=None, **kwargs):
        # breakpoint()
        # completions = [c if c is not None else "" for c in complet]
        if completions.count("") > 0:
            print("weird", completions.count(""))
        # print(list(zip(completions, solution)))
        rewards = [float(completions[i] == solution[i])*10 for i in range(len(completions))]
        # print(rewards)
        return rewards

    template = "Input:\n{}\nOutput:\n"
    
    def convert_to_rl(row):
        # convert from format of query, positive to prompt, solution
        row['prompt'] = template.format(row['query'])
        row['solution'] = row['positive']
        return row
    rl_dataset = dataset['train'].map(convert_to_rl)

    # breakpoint()
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=em_reward,
        train_dataset=rl_dataset,
        args=GRPOConfig(
            save_steps=10000,
            per_device_train_batch_size=64,
            learning_rate=1e-3,
            max_completion_length=100
        )
    )
    trainer.train()