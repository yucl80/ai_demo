from langchain_community.llms import VLLM

llm = VLLM(
    model="THUDM/chatglm2-6b-int4",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

print(llm.invoke("What is the capital of France ?"))