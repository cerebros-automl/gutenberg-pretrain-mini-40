# Gutenberg Pretrain Mini-Series

This repository is part of a collection of projects aimed at creating **optimal subsets** of the [`swiss-ai/apertus-pretrain-gutenberg`](https://huggingface.co/datasets/swiss-ai/apertus-pretrain-gutenberg) dataset, tailored for training **proof-of-concept pico-scale LLMs** at various context lengths.

## Related Projects

- [gutenberg-pretrain-mini-40](https://github.com/cerebros-automl/gutenberg-pretrain-mini-40)
- [gutenberg-pretrain-mini-96](https://github.com/cerebros-automl/gutenberg-pretrain-mini-96)
- [gutenberg-pretrain-mini-1024](https://github.com/cerebros-automl/gutenberg-pretrain-mini-1024)
- _Several more in development:_
  - Synthetic textbook corpora
  - News article datasets
  - Wikipedia-based datasets
  - Instruction and reasoning fine-tuning datasets

## Objective

We aim to create **high-quality training data** for small-scale language models by focusing on:

1. **Synthetic data generation** distilled from reliable sources.
2. **Intelligent summarization** to fit specific sequence lengths, avoiding naive truncation.
3. **Complete thoughts per sample** — ensuring each training example ends naturally, not just naively truncating sentences to make it fit the sample's length.
4. **Proper grammar and punctuation** — maintaining linguistic quality.
5. **Simple packaging** — samples formatted as `list[str]` or CSV for ease of use.
6. **Hugging Face mirrors** — each dataset will be mirrored on Hugging Face with added metadata.

## Data Pipeline Overview

1. **Load Source Dataset**  
   Pull the base dataset (e.g., `swiss-ai/apertus-pretrain-gutenberg`) using Hugging Face's `datasets` library.

2. **Break Down and Summarize**  
   Split and summarize each sample to fit the target context length using an LLM (e.g., `Qwen3-4B-Instruct`).

3. **Quality Control (QC)**  
   - Check token lengths using appropriate tokenizers.
   - Fix or discard samples that are too long.
   - Ensure grammar, punctuation, and completeness of thoughts.

4. **Consolidate and Export**  
   Combine all cleaned samples into a final CSV file for training.

## Key Scripts

### `0-load-and-subset-apertus-pretrain-gutenberg.py`

This script loads the base dataset, filters entries with text, takes a random sample, and saves it to a CSV.

```bash
python3 0-load-and-subset-apertus-pretrain-gutenberg.py
```

Output File:
`swiss-ai-apertus-pretrain-gutenberg-10k.csv`

You can also clone the subset repo directly:

git clone https://github.com/cerebros-automl/gutenberg-pretrain-mini-40.git

process_batch_of_samples.py

Processes text samples using an LLM to summarize and clean them to fit a specific token budget. It also includes QC logic to validate the output.
Usage Example:

Run in a detached screen session for long-running jobs:

screen -S batch_gutenberg_100 -dm bash -c "python3 process_batch_of_samples.py --min_index 0 --max_index 100 &> batch_logs.txt; exit"

Arguments:

    --min_index: Starting index for processing (default: 0)
    --max_index: Ending index for processing (default: end of dataset)

Output:

Each processed sample is saved as a .py file in the gutenberg-batches/ directory.
Contributing

We welcome contributions! If you'd like to help build out new subsets or improve the pipeline, feel free to open issues or submit pull requests.
License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
Acknowledgments

- Thanks to the Cerebros Team, https://github.com/Aidyn-Lopez, https://github.com/Thunderblok, https://github.com/sashakolpakov, https://github.com/Shohail-Ismail, Jeffly, Jennifer, and others who have contributed to the Cerebros project throughout times.
- Thanks to Swiss AI Lab for the original Gutenberg dataset.
- Powered by llama.cpp and Hugging Face.




