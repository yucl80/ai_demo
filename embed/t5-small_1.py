from transformers import T5Model, T5Tokenizer
from sentence_transformers.util import cos_sim
# Load model and tokenizer
model = T5Model.from_pretrained("google-t5/t5-base")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

def emb(input_text):
    # Tokenize input text
    # input_text = "Your input sentence here."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Forward pass through encoder
    output = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True)

    # Get last hidden states
    embeddings = output.last_hidden_state
    return embeddings



a = emb( 'return maximum value')[0][0]
b = emb('def f(a,b): if a>b: return a else return b')[0][0]
c = emb('def f(x,y): if x<y: return y else return x')[0][0]
print(a)
print(cos_sim(a, b))
print(cos_sim(a, c))
print(cos_sim(b, c))