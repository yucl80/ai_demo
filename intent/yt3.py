from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练的BERT模型和tokenizer
model_name = 'shibing624/text2vec-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 定义意图分类器
intent_labels = ['打招呼', '编程', '查询数据库','聊天']

# 用户输入问题
user_input = input("请输入您的问题：")

# 对用户输入进行编码和分类
inputs = tokenizer.encode_plus(
    user_input,
    add_special_tokens=True,
    return_tensors='pt'
)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 使用预训练的BERT模型进行意图分类
outputs = model(input_ids, attention_mask=attention_mask)
intent_index = outputs.logits.argmax().item()
intent = intent_labels[intent_index]

print("用户意图：", intent)