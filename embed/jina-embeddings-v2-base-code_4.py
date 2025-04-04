import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig

# Load tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-code')
config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v2-base-code')

# Tokenize input
input_text = tokenizer('sample text', return_tensors='np')

# ONNX session
model_path = 'D:\\llm\\embed\\jina-embedddings-v2-base-code-onnx/model.onnx'
session = onnxruntime.InferenceSession(model_path)

# Prepare inputs for ONNX model
task_type = 'text-matching'
task_id = np.array(config.lora_adaptations.index(task_type), dtype=np.int64)
inputs = {
    'input_ids': input_text['input_ids'],
    'attention_mask': input_text['attention_mask'],
    'task_id': task_id
}

# Run model
outputs = session.run(None, inputs)
