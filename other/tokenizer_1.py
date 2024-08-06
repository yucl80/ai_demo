from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

model_name = "THUDM/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
tokenizer.tokenize
print(len(inputs))
print(inputs)

