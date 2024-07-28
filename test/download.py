# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
# tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

# from datasets import load_dataset

# ds = load_dataset("openai/gsm8k", "main")