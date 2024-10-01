import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCustomTasks
from sentence_transformers.util import cos_sim
# Load tokenizer from HuggingFace for BERT model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

onnx_model_path = 'D:\llm\jina-embeddings-onnx-o2'
# onnx_model_path ="D:\\llm\\bge-m3-onnx"
# onnx_model_path ="D:\\llm\\bge-base-en-onnx"
# onnx_model_path ="D:\\llm\\all12"

tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

model = ORTModelForCustomTasks.from_pretrained(onnx_model_path)

# Function to process a batch of texts and generate embeddings
def get_embeddings_batch(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors='np', padding=True, truncation=True, max_length=128)

    # ONNX expects input ids, attention masks, and token type ids
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # token_type_ids = inputs['token_type_ids']

    # Prepare inputs for ONNX session
    ort_inputs = {
        'input_ids': np.array(input_ids, dtype=np.int64),
        'attention_mask': np.array(attention_mask, dtype=np.int64),
        # 'token_type_ids': np.array(token_type_ids, dtype=np.int64)
    }
   
    ort_outputs = model.forward(**inputs)["sentence_embedding"]
    # ort_outputs = model.forward(**inputs)["last_hidden_state"]
    # print(ort_outputs)
    
    # ["sentence_embedding"]
    
    return ort_outputs

# Example batch of texts
texts = [
    "This is a test sentence.",
    "ONNX is a great tool for optimization.",
    "Batch processing with ONNX is fast and efficient."
]

# Get embeddings for the batch
embeddings = get_embeddings_batch(texts)

# Output the shape of embeddings
print("Embeddings shape:", embeddings.shape)  # Expecting shape (batch_size, sequence_length, hidden_size)

embeddings = get_embeddings_batch([
    'How do I access the index while iterating over a sequence with a for loop?',
    '# for idx, x in enumerate(xs):\n    print(idx, x)',
])
print(cos_sim(embeddings[0], embeddings[1]))

embeddings = get_embeddings_batch([
    'calculate maximum value',
    'def f(a,b): if a>b: return a else return b',
    "def f(a,b): if a<b: return a else return b"
])
print(cos_sim(embeddings[0], embeddings[1]))
print(cos_sim(embeddings[0], embeddings[2]))
print(cos_sim(embeddings[1], embeddings[2]))