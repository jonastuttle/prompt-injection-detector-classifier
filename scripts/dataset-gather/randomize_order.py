# This script randomizes the order of the entries in the JSONL file for finetuning purposes.
# Change pathing to your own.

import pandas as pd

df = pd.read_json('path/path/dataset_merged.jsonl', lines=True)

# shuffle entries
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_json('path/path/prompt_injection_dataset_final.jsonl', orient='records', lines=True)