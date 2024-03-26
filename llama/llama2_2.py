from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama")

print(llm("AI is going to"))