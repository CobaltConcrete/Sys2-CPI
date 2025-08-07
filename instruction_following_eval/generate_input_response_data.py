import os
import json
import argparse
from unsloth import FastLanguageModel
from transformers import TextStreamer

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Generate model responses for input prompts.")
parser.add_argument("-step", type=int, required=True, help="Checkpoint step to load (e.g. 20, 40, ...)")
parser.add_argument("-ft_dataset_type", type=str, required=True, help="Fine-tuning dataset type (e.g. cheating, naive, etc.)")
args = parser.parse_args()

step = args.step
FT_DATASET_TYPE = args.ft_dataset_type

# -----------------------------
# File and directory paths
# -----------------------------
output_dir = f"/home/r13qingrong/Projects/DSO/instruction_following_eval/data/{FT_DATASET_TYPE}"
os.makedirs(output_dir, exist_ok=True)

INPUT_JSONL_FILE_PATH = "/home/r13qingrong/Projects/DSO/instruction_following_eval/data/input_data.jsonl"
OUTPUT_JSONL_FILE_PATH = f"{output_dir}/input_response_data_unsloth_Qwen3-4B_{FT_DATASET_TYPE}-{step}.jsonl"

model_dir = f"/home/r13qingrong/Projects/Sys2-CPI/unsloth/notebooks/results/qwen3-4b-{FT_DATASET_TYPE}-finetuned_gas4_wus5_lr2e-4_ls1_optim-adamw8bit_wd0_01_lrsched-linear_seed3407/checkpoint-{step}"

# -----------------------------
# Load model and tokenizer
# -----------------------------
max_seq_length = 2048
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# -----------------------------
# Generate responses
# -----------------------------
with open(INPUT_JSONL_FILE_PATH, 'r', encoding='utf-8') as infile, \
     open(OUTPUT_JSONL_FILE_PATH, 'w', encoding='utf-8') as outfile:

    for line in infile:
        data = json.loads(line)
        prompt = data.get("prompt", "")
        chat = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            top_k=30
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Write prompt and response to output JSONL file
        json.dump({
            "prompt": prompt,
            "response": decoded_output
        }, outfile)
        outfile.write("\n")
