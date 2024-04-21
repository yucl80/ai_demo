from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
chat_handler = Llava15ChatHandler(clip_model_path="/home/test/llm-models/mmproj-model-f16.gguf")
llm = Llama(
    model_path="/home/test/llm-models/ggml-model-q4_k.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
    logits_all=True,  # needed to make llava work
)
response = llm.create_chat_completion(
    messages=[
        {"role": "system",
         "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": "https://img0.baidu.com/it/u=3795936406,3301328154&fm=253&fmt=auto&app=138&f=JPEG?w=600&h=400"}},
                {"type": "text", "text": "Describe this image in detail please."}
            ]
        }
    ]
)
print(response)


# import base64

# def image_to_base64_data_uri(file_path):
#     with open(file_path, "rb") as img_file:
#         base64_data = base64.b64encode(img_file.read()).decode('utf-8')
#         return f"data:image/png;base64,{base64_data}"

# # Replace 'file_path.png' with the actual path to your PNG file
# file_path = 'file_path.png'
# data_uri = image_to_base64_data_uri(file_path)

# messages = [
#     {"role": "system", "content": "You are an assistant who perfectly describes images."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": data_uri }},
#             {"type" : "text", "text": "Describe this image in detail please."}
#         ]
#     }
# ]