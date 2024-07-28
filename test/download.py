# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")