from sentence_transformers import SentenceTransformer, util
import time
INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

# Define queries and keys
queries = ["I like you", "I love you","我想你了"]
keys = ["我喜欢你", "我不喜欢你","我爱你","我讨厌你","I don't love you","I hate you","I don't like you"]

# Load model
#model = SentenceTransformer('BAAI/llm-embedder', device="cpu")
model = SentenceTransformer('BAAI/bge-m3', device="cpu")

#model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device="cpu")

# Add instructions for specific task (qa, icl, chat, lrlm, tool, convsearch)
instruction = INSTRUCTIONS["icl"]
queries2 = [instruction["query"] + query for query in queries]
keys2 = [instruction["key"] + key for key in keys]

# Encode
query_embeddings = model.encode(queries2)
key_embeddings = model.encode(keys2)

similarity = query_embeddings @ key_embeddings.T
print(similarity)

start = time.time()
#question_embedding2 = model.encode("I love you")
question_embedding2 = model.encode(instruction["query"] +"I love you")
end = time.time()
print("Time:", end - start) 

correct_hits = util.semantic_search(question_embedding2, key_embeddings, top_k=5)[0]
correct_hits_ids = set([hit["corpus_id"] for hit in correct_hits])
print(correct_hits_ids)
for hit in correct_hits:
        print(keys[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
