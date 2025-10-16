
"""
Takes a subset of 
"""

from llama_cpp import Llama
import gc
import os
import pandas as pd
import numpy as np
import argparse
import sys

MAX_TOKENS = 11_500

# Initialize the LLM
llm = Llama.from_pretrained(
    repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen3-4B-Instruct-2507-F16.gguf",
    n_ctx=MAX_TOKENS,
)
gc.collect()


def send_single_turn_instruction(prompt_0: str) -> str:
    messages = [{"role": "user", "content": prompt_0}]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1000,
        temperature=0.65,
        top_p=0.95,
        top_k=45,
        repeat_penalty=1.2
    )
    response_text = response["choices"][0]["message"]["content"]
    gc.collect()
    return response_text

SEQUENCE_LENGTH_TARGET = 35

QC_SCRIPT = """


from transformers import AutoTokenizer
tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

print(f"Number of Samples: {len(samples)}")
tokens = [tokenizer(sample)['input_ids'] for sample in samples]


import numpy as np
sample_word_counts = np.array([len(a) for a in tokens])
print(f"max len: {sample_word_counts.max()}")
print(f"min len: {sample_word_counts.min()}")
print(f"mean len: {sample_word_counts.mean()}")
print(f"len std: {sample_word_counts.std()}")
print(f"num over token count: {np.array(sample_word_counts > 40).sum()}")
# print("NEW SAMPLE:")
# samples_np = np.array(samples, dtype=object)
# print([str(x) for x in samples_np[sample_word_counts <= 40].tolist()])

"""

def process_samples(sample: str) -> str:
    PROMPT = f"""# Please process this text sample according to these rules and return a list of Python strings derived from it that meets these constraints:

* Context length summarization rather than truncation to proof-of-concept scale {SEQUENCE_LENGTH_TARGET} tokens
* Strip anything that is not English prose—citations, URLs, line wraps, labels, verse numbers, page numbers footnotes, stray Unicode, etc.
* Ensuring that samples begin with proper capitalization, end with correct punctuation and a natural end of paragraph, not just naively truncating sequential sentences in the text as separate samples that end “mid - paragraph” … which would have the undesired effect of encouraging the model to write in a verbose format beyond the context window and terminate its writings mid paragraph / throw a stop token without expressing a complete thought.
* Package as a Python list[str] named `samples`.
* PLEASE DO NOT MAKE ANY COMMENTS ABOUT THIS SUCH AS "Here is the text samples you requested:" ... JUST return the samples as requested.

# Here is the text to process:

{sample}
"""
    processed_samples = send_single_turn_instruction(prompt_0=PROMPT)
    processed_samples_with_qc = processed_samples + QC_SCRIPT
    return processed_samples_with_qc


def main(min_index: int = 0, max_index: int = None):
    # Load data
    all_samples_file = "swiss-ai-apertus-pretrain-gutenberg-10k.csv"
    try:
        raw_samples = pd.read_csv(all_samples_file)
    except FileNotFoundError:
        print(f"Error: File '{all_samples_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    all_samples_list = raw_samples["text"].tolist()
    
    # Handle max_index
    if max_index is None:
        max_index = len(all_samples_list)
    else:
        max_index = min(max_index, len(all_samples_list))
    
    # Validate indices
    if min_index < 0 or min_index >= len(all_samples_list):
        print(f"Error: min_index {min_index} is out of range.")
        sys.exit(1)
    
    if max_index <= min_index:
        print(f"Error: max_index {max_index} must be greater than min_index {min_index}.")
        sys.exit(1)
    
    output_dir = "gutenberg-batches"
    os.makedirs(output_dir, exist_ok=True)
    # Process samples in range
    for i in range(min_index, max_index):
        sample = all_samples_list[i]
        outfile = f"{output_dir}/gutenberg_step1-{i}.py"
        try:
            out_samples = process_samples(sample)
            with open(outfile, "w") as file:
                file.write(out_samples)
            print(f"Processed and saved sample {i} to {outfile}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text samples with LLM")
    parser.add_argument("--min_index", type=int, default=0, help="Minimum sample index to process (default: 0)")
    parser.add_argument("--max_index", type=int, default=None, help="Maximum sample index to process (default: all samples)")

    args = parser.parse_args()
    main(args.min_index, args.max_index)

