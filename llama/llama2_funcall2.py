from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
#llm = Llama(model_path="/home/test/src/llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf", chat_format="chatml-function-calling")
llm = Llama.from_pretrained( 
  filename="functionary-small-v2.4.Q4_0.gguf",
  chat_format="functionary-v2",
  tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
  repo_id="meetkai/functionary-small-v2.4-GGUF"
)