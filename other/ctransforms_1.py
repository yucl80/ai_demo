from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import login
login(token="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")

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral", gpu_layers=0, hf=True
)
tokenizer = AutoTokenizer.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2', use_fast=True
)

messages = [
    {"role": "user", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "assistant", "content": "How many helicopters can a human eat in one sitting?"},
   
]

print(tokenizer.chat_template)

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
