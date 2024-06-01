
from transformers import PreTrainedTokenizerFast, BertTokenizer
from transformers import (AutoModelForCausalLM, LlamaForCausalLM,PreTrainedTokenizerFast,
                          LlamaTokenizerFast,AutoTokenizer)

#tokenizer = LlamaTokenizerFast.from_pretrained( "meetkai/functionary-small-v2.5-GGUF", legacy=False)

tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-small-v2.4", trust_remote_code=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
messages = [{"role": "user", "content": "What is the weather in Istanbul and Singapore respectively?"}]

final_prompt = tokenizer.apply_chat_template(messages, tools, add_generation_prompt=True, tokenize=False)
tokenizer.padding_side = "left"

print(final_prompt)

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
tokenized_chat = tokenizer.apply_chat_template(messages, tools,  tokenize=False, add_generation_prompt=True, return_tensors='pt')
print(tokenized_chat)
# print(tokenizer.decode(tokenized_chat[0]))

#print(fast_tokenizer)

#print(tokenizer.convert_ids_to_tokens(template))

