## Dataset

The original dataset [Safe Guard Prompt Injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) contains the following format:

- text - the raw prompt that is to be given to a model.
- label - classified by 0 or 1, this indicates whether the text/prompt is not a prompt injection or is a prompt injection, respectively.

This dataset is nearly perfect, but lacks classification on the type of prompt injection an entry is, thus we have automated the process of classifying it ourself, then manually reviewing the entries. The result is a dataset that now contains a third category:

- injection_type - simply classifies the type of prompt injection the entry is, given that the prompt is in fact a prompt injection.


**Our Dataset**

Due to the fine-tuning method of multi-class classification, the dataset needs to be adapted further. This is simply done by removing the "label" feature, and adding all 0's from that feature to have a new entry called "benign". The final results is this:

- text - the raw prompt that is to be given to a model
- label - marks the type of prompt injection:

```json
{

"Adversarial Example": 0,

"Harmful Request": 1,

"Indirect Manipulation": 2,

"Instruction Override": 3,

"Jailbreak Attempt": 4,

"Other": 5,

"Prompt Leaking": 6,

"Role Impersonation": 7,

"benign": 8

}
```

- 8236 entries
- jsonl format


### Methodology

**Model**
The model chosen is one that is already trained for prompt injection detection: [Prompt Injection Classifier](https://huggingface.co/xTRam1/safe-guard-classifier)

The original description of this model read:

*"*
*We formulated the prompt injection detector problem as a classification problem and trained our own language model to detect whether a given user prompt is an attack or safe.*
*"*
This model misses the classifications of the type, which is where our unique findings come in...

**Multi-Class Classification**
- Combined both injection detection and classification into one classification task.
- model learns to predict *the type* of injection for each entry, with "benign" being treated as a class.

The approach is simple as it simple utilizes the model being given the original prompt and the label which is the classification type. In simple terms, the model learns to "match" prompts of similar types to their respective prompt injection type over time. For Example:

- after given enough examples of seeing "Ignore all previous instructions...", the model will begin to see this as a pattern to the label "Instruction Override" and will "learn" that a prompt with that phasing is highly likely to be that injection type.

**Tools/Framework/Environment**

- Transformers - AutoTokenizer  for tokenizing prompts and AutoModelForSequenceClassification for classification head.
- Parameter Arguments - many were tested, but the current model used:
	- epochs: 10
	- train batch: 8
	- learning rate: 1e-5

- Deepspeed - config file used for acceleration and compute optimizations (hardware constraints
- Hardware: 4x 2080 Ti's

**Testing**

To ensure fair testing, 10% of the training dataset was set aside strictly for testing. That is, the entries used in testing are not used for the training phase, ensuring the model will not have seen these prompts/labels before.

After fine-tuning was complete, the model is simply prompted the 10% unseen data, to which the model consistently performs between a 96%-100% accuracy. Never less than 96% and usually around 97% or 98% accuracy.


### Fin
The link to the glorious model is [here!](https://huggingface.co/jonastuttle/NobleGuardClassifier)