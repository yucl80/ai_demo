import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCustomTasks

# Load tokenizer from HuggingFace for BERT model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained("D:\\llm\\jina")

# Load ONNX model
session = ort.InferenceSession( 'D:\\llm\\jina\\model.onnx')


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

    # Run inference
    ort_outputs = session.run(None, ort_inputs)    

    # The model's output should contain the embeddings (e.g., last hidden state)
    embeddings = ort_outputs.tolist()  # Depending on your model, this index might vary
    return embeddings

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
