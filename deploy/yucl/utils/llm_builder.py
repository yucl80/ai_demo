from langchain_openai import ChatOpenAI
import time
import jwt
import os

def get_api_token(): 
    api_key = os.environ["ZHIPUAI_API_KEY"] 
    exp_seconds= 100
    try:
        id, secret = api_key.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)
 
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
 
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def get_api_key():
    return os.environ["ZHIPUAI_API_KEY"]

def create_llm(model="glm-3-turbo",temperature=0.01, timeout=180, max_retries=0,streaming=False):
    """create a language model using the OpenAI API."""
    return ChatOpenAI(
        model_name=model,
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key= "EMPTY_KEY",
        streaming= streaming,
        temperature= temperature,        
        request_timeout= timeout,
        max_retries= max_retries,
    ) 
   

def create_glm4(temperature=0.01, timeout=180):
    """create a glm-4 language model using the OpenAI API."""
    return ChatOpenAI(
        model_name="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4/",       
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    ) 

