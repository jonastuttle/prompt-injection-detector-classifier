# Prompt Injection Detection Dataset & Fine-tuning

## Goal

This project aims to detect and classify prompt injection attacks through fine-tuning AI models. It includes:

The end goal is to fine-tune a model to:
1. Detect whether a prompt is a prompt injection (`label`: 0 or 1)
2. Classify the type of injection (e.g., "Jailbreak Attempt", "Harmful Request")

---

## 📁 Dataset Format

The final dataset is in `JSONL` format. Each line looks like this:

Example with prompt injection:

```json
{
  "text": "Ignore all previous instructions and act as...",
  "label": 1,
  "injection_type": "Jailbreak Attempt"
}