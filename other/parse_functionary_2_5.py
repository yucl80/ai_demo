import uuid
response = {'id': 'chatcmpl-3bc7d22d-b5dd-41c2-900f-01617805b3f3', 'object': 'chat.completion', 'created': 1716879542, 'model': '/home/test/.cache/huggingface/hub/models--meetkai--functionary-small-v2.5-GGUF/snapshots/85e1e6264ec8984a3014a2b1a7a16f3f997a45a3/./functionary-small-v2.5.Q4_0.gguf', 'choices': [{'index': 0, 'logprobs': None, 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_2zF6JuQUU2PiUgz176paL4Ga', 'type': 'function', 'function': {'name': 'functions.get_current_weather\n{"location": "San Francisco"}<|reserved_special_token_249|>get_current_weather\n{"location": "Tokyo"}<|reserved_special_token_249|>get_current_weather\n{"location": "Paris"}', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 209, 'completion_tokens': 36, 'total_tokens': 209}}

for choice in response["choices"]:
        message_content = choice["message"]["content"]
        if choice["finish_reason"] == "tool_calls":
            tool_calls= []
            for tool_call in choice["message"]["tool_calls"]:
                function_name = tool_call["function"]["name"]
                if function_name.startswith('functions.'):
                    function_calls = function_name.split('functions.')[1]
                    functions = function_calls.split('<|reserved_special_token_249|>')
                    for func in functions:
                        fcall = func.split('\n')
                        print(fcall)                       
                        tool_calls.append({
                            "id": "tool_call_" + uuid.uuid4().hex,
                            "type": "function",
                            "function": {
                                "name":  fcall[0],
                                "arguments": fcall[1], 
                            }
                        })                       

            if message_content is not None and  message_content.startswith('<|reserved_special_token_249|>'):
                tool_calls=[]
                if message_content.startswith('<|reserved_special_token_249|>'):
                    function_name = message_content.split('\n')[0].split('<|reserved_special_token_249|>')[1]
                    function_args = message_content.split('\n')[1]
                    tool_calls.append({
                        "id": "tool_call_" + uuid.uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_args, 
                        }
                    })
                    choice["message"]["tool_calls"] = tool_calls
                    choice["finish_reason"]="tool_calls"
            
            print(tool_calls)
        
#print("response:", end=" ")
#print(response)


# str = '<|reserved_special_token_249|>getStockPrice\n{"stockName": "GOOGL"}'

# str1 = None

# if  '<|reserved_special_token_249|>' in str1:
#     print("yes")

# if str.startswith('<|reserved_special_token_249|>'):
#     function_name = str.split('\n')[0].split('<|reserved_special_token_249|>')[1]
#     function_args = str.split('\n')[1]
#     print(function_name)
#     print(function_args)




#  tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-small-v2.5", trust_remote_code=True)
#     user_prompt = tokenizer.apply_chat_template(body.messages,body.tools, tokenize=False, add_generation_prompt=True, return_tensors='pt')
#     print(user_prompt)
#     # messages = []
#     # for msg in body.messages:
#     #     role = msg["role"]
#     #     if role not in ["user", "system", "tool"]:
#     #         role = "user"
#     #     messages.append({"role": role, "content": msg["content"]})
#     # print("call functionary")
#     # response = llama(
#     #     user_prompt,
#     #     stop=["</s>"],
#     #     temperature=body.temperature,
#     #     top_p=body.top_p,
#     #     max_tokens=body.max_tokens,
#     # )
#     # print(response)
#     # print(body.messages)
#     tokenizer.padding_side = "left"