from llama_cpp import Llama
from transformers import AutoTokenizer
llm = Llama(model_path="/home/test/llm-models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  n_threads=12,
            verbose=False)

messages = [    
           
    {"role": "user", "content": "How many helicopters can a human eat in one sitting ?"},
   
]
output = llm.create_chat_completion(messages=messages

                                    )
print(output['choices'][0]['message']['content'])

tokenizer = AutoTokenizer.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2', use_fast=True
)

print(tokenizer.chat_template)

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_tensors='pt')

print(tokenized_chat)

output = llm(tokenized_chat,
             max_tokens=150,
             echo=False,)
print(output['choices'][0]['text'])

prompt_template = '''[INST] <<SYS>>
You are a friendly chatbot who always responds in the style of a pirate
<</SYS>>
{prompt}[/INST]
'''

output = llm(prompt_template.format(prompt='Name the planets in our solar system'),
             max_tokens=150,
             echo=False,)
print(output['choices'][0]['text'])



