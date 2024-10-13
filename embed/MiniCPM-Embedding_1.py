
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "openbmb/MiniCPM-Embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to("cpu")
model.eval()

# 由于在 `model.forward` 中缩放了最终隐层表示，此处的 mean pooling 实际上起到了 weighted mean pooling 的作用
# As we scale hidden states in `model.forward`, mean pooling here actually works as weighted mean pooling
def mean_pooling(hidden, attention_mask):
    s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
    d = attention_mask.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(input_texts):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to("cpu")
    
    outputs = model(**batch_dict)
    attention_mask = batch_dict["attention_mask"]
    hidden = outputs.last_hidden_state

    reps = mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

queries = ["get the maximum value",'获取最大数']
passages = [ 'def f(a,b): if a>b: return a else return b',    'def f(x,y): if x<y: return y else return x']


INSTRUCTION = "Query: "
queries = [INSTRUCTION + query for query in queries]

embeddings_query = encode(queries)
embeddings_doc = encode(passages)

scores = (embeddings_query @ embeddings_doc.T)
print(scores.tolist())  # [[0.3535913825035095, 0.18596848845481873]]


# import torch
# from sentence_transformers import SentenceTransformer

# model_name = "openbmb/MiniCPM-Embedding"
# model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={ "torch_dtype": torch.double})
