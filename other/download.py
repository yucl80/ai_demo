from huggingface_hub import snapshot_download
model_id = "THUDM/chatglm3-6b"
snapshot_download(repo_id=model_id, local_dir="home/test/tmp/chatglm3-6b",max_workers=2,
                  local_dir_use_symlinks=False, revision="main",resume_download=True)


# python llama.cpp/convert.py vicuna-hf \
#   --outfile vicuna-13b-v1.5.gguf \
#   --outtype q8_0

# cd llama.cpp/build/bin && \
#    ./quantize ./models/Llama-2-7b-chat-hf/ggml-model-f16.gguf ./models/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf q4_0