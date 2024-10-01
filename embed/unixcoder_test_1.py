import torch
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

# Encode maximum function
func = "def f(a,b): if a>b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,max_func_embedding = model(source_ids)

# Encode minimum function
func = "def f(a,b): if a<b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,min_func_embedding = model(source_ids)

# Encode NL
nl = "return maximum value"
tokens_ids = model.tokenize([nl],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,nl_embedding = model(source_ids)

print(max_func_embedding.shape)
print(max_func_embedding)

norm_max_func_embedding = torch.nn.functional.normalize(max_func_embedding, p=2, dim=1)
norm_min_func_embedding = torch.nn.functional.normalize(min_func_embedding, p=2, dim=1)
norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

max_func_nl_similarity = torch.einsum("ac,bc->ab",norm_max_func_embedding,norm_nl_embedding)
min_func_nl_similarity = torch.einsum("ac,bc->ab",norm_min_func_embedding,norm_nl_embedding)

print(max_func_nl_similarity)
print(min_func_nl_similarity)

context = """
def <mask0>(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])


context = """
# <mask0>
def write_json(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])
