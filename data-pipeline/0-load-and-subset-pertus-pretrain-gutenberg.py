"""

Simply loads the dataset, selects 10_000 text samples by quasi random selection, and saves the result as a CSV.

"""

from datasets import load_dataset

# Load dataset in streaming mode
ds = load_dataset("swiss-ai/apertus-pretrain-gutenberg", streaming=True, split="train")

# Filter examples that have the 'text' field
filtered_ds = ds.filter(lambda example: 'text' in example)

# Shuffle and take 10,000 samples
random_sample = filtered_ds.shuffle(seed=42).take(10000)

# Convert to pandas DataFrame
import pandas as pd

texts = [example['text'] for example in random_sample]
df = pd.DataFrame({'text': texts})

# Save to CSV
df.to_csv('swiss-ai-apertus-pretrain-gutenberg-10k.csv', index=False)

print(f"Successfully wrote {len(df)} rows to swiss-ai-apertus-pretrain-gutenberg-10k.csv")
