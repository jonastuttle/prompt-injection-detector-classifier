import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100, help="Number of prompts to test")
args = parser.parse_args()

with open("./datasets/injection_final.jsonl", "r") as f:
    dataset = [json.loads(line.strip()) for line in f]

with open("./artifacts/label2id.json") as f:
    label2id = json.load(f)
with open("/home//Security-Project/artifacts/id2label.json") as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}

all_labels = sorted(id2label.keys())

model_path = "./scripts/finetune/checkpoints/checkpoint-2320"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def classify_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
        return pred_id, confidence

sampled = random.sample(dataset, min(args.n, len(dataset)))

true_labels = []
pred_labels = []

print(f"\n🔍 Evaluating {len(sampled)} prompts...\n")

for i, entry in enumerate(sampled):
    text = entry["text"]
    true_label_str = entry["label"]
    true_label = label2id[true_label_str]

    pred_label, conf = classify_prompt(text)

    true_labels.append(true_label)
    pred_labels.append(pred_label)

    print(f"{i+1}. Prompt: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"   Predicted: {id2label[pred_label]} (conf: {round(conf, 4)}) | True: {true_label_str}")
    print("")

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="weighted")

print("=======Evaluation Summary=======")
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