import os
import csv
import nltk
from absl import app, flags, logging
from instruction_following_eval import evaluation_lib

nltk.download('punkt_tab')

_INPUT_DATA = flags.DEFINE_string("input_data", None, "Path to input_data.jsonl", required=True)
_FT_DATASET_TYPE = flags.DEFINE_string("ft_dataset_type", None, "Finetuning dataset type (e.g., paraphrase, implication, qa, etc.)", required=True)


def evaluate_file(input_data_path, response_file_path):
    inputs = evaluation_lib.read_prompt_list(input_data_path)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(response_file_path)

    def compute_accuracy(outputs):
        prompt_correct = sum(o.follow_all_instructions for o in outputs)
        prompt_total = len(outputs)
        instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
        instruction_total = sum(len(o.follow_instruction_list) for o in outputs)
        return (
            prompt_correct / prompt_total if prompt_total else 0.0,
            instruction_correct / instruction_total if instruction_total else 0.0
        )

    # Strict
    strict_outputs = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    strict_prompt_acc, strict_instr_acc = compute_accuracy(strict_outputs)

    # Loose
    loose_outputs = [evaluation_lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]
    loose_prompt_acc, loose_instr_acc = compute_accuracy(loose_outputs)

    return strict_prompt_acc, strict_instr_acc, loose_prompt_acc, loose_instr_acc


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    input_data_path = _INPUT_DATA.value
    dataset_type = _FT_DATASET_TYPE.value

    # input_response_dir = os.path.join("instruction_following_eval", "data", "cleaned", dataset_type)
    input_response_dir = os.path.join("instruction_following_eval", "data", dataset_type, "temp_0_6")
    output_results_file = os.path.join("instruction_following_eval", "data", "results", f"{dataset_type}_results.csv")

    os.makedirs(os.path.dirname(output_results_file), exist_ok=True)

    with open(output_results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Strict Prompt Acc", "Strict Instr Acc", "Loose Prompt Acc", "Loose Instr Acc"])

        for fname in sorted(os.listdir(input_response_dir)):
            if fname.endswith(".jsonl"):
                response_path = os.path.join(input_response_dir, fname)
                print(f"Evaluating {fname}...")
                try:
                    strict_p, strict_i, loose_p, loose_i = evaluate_file(input_data_path, response_path)
                    writer.writerow([fname, strict_p, strict_i, loose_p, loose_i])
                except KeyError as e:
                    print(f"[ERROR] {fname} - Missing prompt: {e}")
                except Exception as e:
                    print(f"[ERROR] {fname} - {e}")

    print(f"\n✅ All evaluations complete. Results saved to: {output_results_file}")


if __name__ == "__main__":
    app.run(main)
