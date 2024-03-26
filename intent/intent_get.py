import transformers
import torch

model_name = "shibing624/text2vec-base-chinese"
model_name =  "BAAI/bge-m3"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

intent_list = ["打招呼", "再见", "开灯", "关灯", "调亮"]
threshold = 0.6

def get_intent(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    max_probability = max(probabilities)
    if max_probability >= threshold:
        intent = intent_list[probabilities.index(max_probability)]
        return intent
    return None

while True:
    message = input("User: ")
    intent = get_intent(message)
    if intent is not None:
        print("Bot: detected intent is", intent)
    else:
        print("Bot: no matching intent found")
