from transformers import pipeline
import torch

torch.manual_seed(0)
generator = pipeline('text-generation', model = 'openai-community/gpt2')
prompt = "Hello, I'm a language model"

generator(prompt, max_length = 30)