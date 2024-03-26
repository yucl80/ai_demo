import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载BERT模型和tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 定义意图识别函数
def intent_recognition(text):
    # 使用tokenizer对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 使用模型进行预测
    outputs = model(**inputs)
    # 获取预测结果
    logits = outputs.logits
    # 获取预测的意图标签
    predicted_label = torch.argmax(logits, dim=1).item()
    # 返回意图标签
    return predicted_label

# 测试意图识别函数
text = "我想预订一张机票"
intent = intent_recognition(text)
print("意图识别结果:", intent)