# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

# from datasets import load_dataset

# ds = load_dataset("openai/gsm8k", "main")