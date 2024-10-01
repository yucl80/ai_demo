from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="meetkai/functionary-small-v3.2-GGUF", model_file="D:\\llm\\LMStudio\\meetkai\\functionary-small-v3.2\\functionary-small-v3.2.Q4_0.gguf")

print(llm("AI is going to"))