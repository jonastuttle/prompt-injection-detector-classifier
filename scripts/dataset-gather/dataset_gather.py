# This script gathers a dataset of prompt injections and classifies them using OpenAI's API.

from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import csv
import backoff

client = OpenAI()

# Get dataset it from huggingface
dataset = load_dataset("xTRam1/safe-guard-prompt-injection")
df = pd.DataFrame(dataset['train'])

# Filters only prompt injection entries
injection_df = df[df['label'] == 1].copy().reset_index(drop=True)

# Retry failed calls
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def classify_injection(prompt_text):
    system_prompt = """You are an expert in AI safety. 
    Given a user-submitted prompt that attempts a prompt injection, classify it into one of the following categories:

    1. Instruction Override: Explicitly disregards or overrides AI's instructions.
    2. Role Impersonation: Asks the AI to pretend to be a different person or identity.
    3. Jailbreak Attempt: Explicitly attempts to bypass safety measures or restrictions.
    4. Harmful Request: Direct request for dangerous, unethical, illegal, or unsafe information.
    5. Indirect Manipulation: Subtle or indirect methods of manipulating the model’s responses.
    6. Prompt Leaking: Attempts to reveal internal AI instructions or prompt details.
    7. Adversarial Example: Crafted to cause erratic or unexpected model behavior.
    8. Other: Does not clearly match any above category.

    Respond ONLY with the exact category name.
    """

    user_prompt = f"Classify this prompt injection:\n\n{prompt_text}\n\nCategory:"

    response = client.responses.create(
        model="gpt-4o",
        input=user_prompt,
        instructions=system_prompt,
        temperature=0.0,
        max_output_tokens=32
    )

    return response.output[0].content[0].text.strip()

# Classify each prompt injection
injection_types = []
for i, prompt in enumerate(tqdm(injection_df['text'], desc="Classifying Injections")):
    try:
        category = classify_injection(prompt)
        injection_types.append(category)
        if i % 50 == 0:
            print(f"✔️ Processed {i}/{len(injection_df)} entries")
        time.sleep(1)  # rate limit protection
    except Exception as e:
        print("Error: ", e)
        injection_types.append("Error")

# Add injection type
injection_df['injection_type'] = injection_types

injection_df.to_csv("classified_injection_dataset.csv", index=False, quoting=csv.QUOTE_ALL)

print("Done!")
