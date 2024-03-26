#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import json
from flask import Flask, request
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tokenizer = AutoTokenizer.from_pretrained(r"chatglm-6b", trust_remote_code=True,
                                          revision="main")
model = AutoModel.from_pretrained(r"chatglm-6b", trust_remote_code=True,
                                  revision="main").half().quantize(4).cuda()
model = model.cuda()
model = model.eval()
app = Flask(import_name=__name__)

def predict(input_string, history):
    if history is None:
        history = []
    try:
        response, history = model.chat(tokenizer, input_string, history)
        return {"msg": "success", "code": 200, "response": response}
    except Exception as error:
        return {"msg": "error", "code": 500, "response": error}

@app.route("/chat_with_history", methods=["POST", "GET"])
def chat_with_history():
    data = json.loads(request.data)
    input_text = data['input_text']
    history = data.get("history", None)
    if history is not None:
        history = json.loads(history)
    return predict(input_text, history)

if __name__ == '__main__':
    app.run(port=12345, debug=True, host='0.0.0.0')
