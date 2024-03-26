import transformers
import torch

tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

intent_list = ["greeting", "goodbye", "thank you", "question", ...]
threshold = 0.8

def get_intent(message):
    inputs = tokenizer(message, return_tensors="pt")
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