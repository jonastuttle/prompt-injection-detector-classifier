import json
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate

# paths
model_name = "xTRam1/safe-guard-classifier"
jsonl_path = "./datasets/injection_final.jsonl"
train_path = "./datasets/train_split.jsonl"
eval_path = "./datasets/heldout_eval_split.jsonl"
ds_config_path = "ds_config.json"
output_dir = "./scripts/finetune/output/checkpoints"

# load label mappings
with open("./artifacts/label2id.json") as f:
    label2id = json.load(f)
id2label = {int(v): k for k, v in label2id.items()}
num_labels = len(label2id)

raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example
raw_dataset = raw_dataset.map(encode_labels)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

raw_dataset = raw_dataset.map(tokenize, batched=True)
raw_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# split the dataset into train and eval sets
random.seed(42)
split = raw_dataset.train_test_split(test_size=0.1)
split["train"].to_json(train_path)
split["test"].to_json(eval_path)

train_dataset = split["train"]
eval_dataset = split["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# the bread and butter!
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,
    deepspeed=ds_config_path,
    fp16=True,
    warmup_ratio=0.1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()