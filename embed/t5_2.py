from transformers import T5Tokenizer, T5EncoderModel
import torch
from sentence_transformers.util import cos_sim

# model_id="Salesforce/codet5-base"
model_id="google-t5/t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5EncoderModel.from_pretrained(model_id)


def emb(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids    
    with torch.no_grad():
        embeddings = model(input_ids).last_hidden_state
        return embeddings
   



a = emb( 'return maximum value')[0][0]
b = emb('def f(a,b): if a>b: return a else return b')[0][0]
c = emb('def f(x,y): if x<y: return y else return x')[0][0]
print(a)
print(cos_sim(a, b))
print(cos_sim(a, c))
print(cos_sim(b, c))    