# This script merges the classified injection dataset with the non-injection dataset from Hugging Face.
# Change pathing to your own.

import pandas as pd
from datasets import load_dataset

# Load both datasets (change the injection path to your own)
injection_df = pd.read_csv('/path/path/classified_injection_dataset.csv')
dataset = load_dataset('xTRam1/safe-guard-prompt-injection', split='train')
non_injection_df = pd.DataFrame(dataset)

# Filter out non-injection entries and label 'None' for the new injection type
non_injection_df = non_injection_df[non_injection_df['label'] == 0]
non_injection_df['injection_type'] = 'None'

# Combine the datasets and save as JSONL
combined_df = pd.concat([injection_df, non_injection_df], ignore_index=True)
combined_df.to_json('/path/path/dataset_merged.jsonl', orient='records', lines=True)