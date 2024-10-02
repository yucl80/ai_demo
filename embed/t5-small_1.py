from transformers import T5Model, T5Tokenizer

# Load model and tokenizer
model = T5Model.from_pretrained("google-t5/t5-base")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

# Tokenize input text
input_text = "Your input sentence here."
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through encoder
output = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True)

# Get last hidden states
embeddings = output.last_hidden_state

print(embeddings)