
from transformers import PreTrainedTokenizerFast, BertTokenizer
from transformers import (AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaTokenizerFast)
tokenizer = LlamaTokenizerFast.from_pretrained(
    "meetkai/functionary-small-v2.4-GGUF", legacy=False)



# print("begin -----------------")
# print(tokenizer)
# print("end ------------------")

# print(tokenizer.eos_token)
# print(tokenizer.chat_template)

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="/home/test/.cache/huggingface/hub/models--meetkai--functionary-small-v2.4-GGUF/snapshots/a0d171eb78e02a58858c464e278234afbcf85c5c/tokenizer.json")

messages=[
    {'role': 'system', 'content': 'You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.'}, 
    {'role': 'user', 'content': 'what is LangChain?'},
    {'role': 'assistant', 'content': None, 
        'tool_calls': [{'id': 'call_C4df2xd6obyxQeeegMj0otNM', 'type': 'function', 'function': {'name': 'tavily_search_results_json', 'arguments': '"{\\"query\\": \\"\\\\\\"LangChain definition\\\\\\"\\"}"'}
                        }]
    }]
print(tokenizer.chat_template)
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors='pt')
print(tokenized_chat)
# print(tokenizer.decode(tokenized_chat[0]))

#print(fast_tokenizer)

#print(tokenizer.convert_ids_to_tokens(template))

