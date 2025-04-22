from datasets import load_dataset
from transformers import AutoTokenizer
import json

jsonl_path = "./datasets/injection_final.jsonl"
model_name = "xTRam1/safe-guard-classifier"
max_length = 512

dataset = load_dataset("json", data_files=jsonl_path, split="train")

unique_labels = sorted(list(set(example["label"] for example in dataset)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

with open("./artifacts/label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)
with open("./artifacts/id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("Done!")