from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "gorilla-llm/gorilla-openfunctions-v2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    {"role": "system", "content": "system message 2"},
    {"role": "user", "content": "user message 2"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.chat_template)
print(tokenized_chat)

#print(tokenizer.decode(tokenized_chat[0]))