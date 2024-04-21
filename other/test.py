import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "namespace-Pt/activation-beacon-llama2-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)

model = model.eval()

with torch.no_grad():
  # short context
  text = "Tell me about yourself."
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=200)
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Output:       {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

  # reset memory before new generation task
  model.memory.reset()
  text = "怎么防止失眠?."
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=2000)
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Output:       {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

