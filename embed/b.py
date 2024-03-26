from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# 用户输入的对话内容
user_input = "明天下午三点想知道北京的天气怎么样？"

# 使用tokenizer对用户输入进行编码
inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

# 使用BERT模型进行意图识别和命名实体识别
outputs = model(**inputs)
print( outputs)

# 获取答案起始和结束的概率分数
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# 将张量转换为Python列表
start_scores = start_scores.tolist()
end_scores = end_scores.tolist()

# 打印答案起始和结束的概率分数
print("Start Scores:", start_scores)
print("End Scores:", end_scores)

# 解析模型输出，识别意图和关键信息
intent = "查询天气"  # 这里假设直接将意图设为查询天气
date = "明天"  # 这里假设直接将日期设为明天
time = "下午三点"  # 这里假设直接将时间设为下午三点
location = "北京"  # 这里假设直接将地点设为北京

# 根据识别出的意图和关键信息调用相应的服务或程序
if intent == "查询天气":
    # 调用天气查询服务，并传递识别出的日期、时间和地点作为参数
    weather_info = f"查询到{location}的天气情况：晴，温度25摄氏度。"
    print(weather_info)
else:
    # 处理其他意图的逻辑
    pass
