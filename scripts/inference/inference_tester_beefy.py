import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

model_path = "./scripts/finetune/output/checkpoints/checkpoint-2320"
eval_path = "./datasets/heldout_eval_split.jsonl"

with open("./artifacts/label2id.json") as f:
    label2id = json.load(f)
with open("./artifacts/id2label.json") as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}
all_labels = sorted(id2label.keys())

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

dataset = load_dataset("json", data_files=eval_path, split="train")

def classify_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
        return pred_id, confidence

true_labels = []
pred_labels = []

print(f"\n Evaluating {len(dataset)} prompts from held-out eval set...\n")

for i, entry in enumerate(dataset):
    text = entry["text"]
    true_label = entry["label"]

    pred_label, conf = classify_prompt(text)

    true_labels.append(true_label)
    pred_labels.append(pred_label)

    print(f"{i+1}. Prompt: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"    Predicted: {id2label[pred_label]} (conf: {round(conf, 4)}) | True: {id2label[true_label]}")
    print("")

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="weighted")

print("======= Evaluation on Held-Out Data =======")
print(f"Accuracy: {round(accuracy * 100, 2)}%")
print(f"F1 Score: {round(f1, 4)}\n")

print("Classification Report:")
print(classification_report(
    true_labels,
    pred_labels,
    labels=all_labels,
    target_names=[id2label[i] for i in all_labels],
    zero_division=0
))