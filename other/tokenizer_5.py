from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'shenzhi-wang/Llama3-8B-Chinese-Chat', use_fast=True
)

# tokenizer = AutoTokenizer.from_pretrained(
#     'Nexusflow/NexusRaven-V2-13B', use_fast=True
# )

print(tokenizer.chat_template)

messages = [
    {"role": "system", "content": "Hi there!"},
    {"role": "user", "content": "Nice to meet you!"},
    {"role": "assistant", "content": "Can I ask a question?"}
]

results = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(results)

results = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print(results)