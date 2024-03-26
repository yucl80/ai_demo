from modelscope import AutoTokenizer, AutoModel, snapshot_download
model_dir = "ZhipuAI/chatglm3-6b"
#model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
#cuda使用GPU计算, float使用CPU计算
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().float()
#model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).cpu().float()
model = model.quantize(bits=4, parallel_num=12)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)