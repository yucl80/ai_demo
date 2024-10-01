from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("D:/llm/LMStudio/lmstudio-community/Qwen/gte-qwen2-7b-instruct-q4_k_m.gguf", model_type="llm")

#print(llm("AI is going to"))
embeddings = llm.embed("Hello, world!")

print(len(embeddings))

#print(embeddings)

#



#for text in llm("AI is going to", stream=True):
#    print(text, end="", flush=True)
    
#You can load models from Hugging Face Hub directly:

#llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml")
#If a model repo has multiple model files (.bin or .gguf files), specify a model file using:

#llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", model_file="ggml-model.bin")