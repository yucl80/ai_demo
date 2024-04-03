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

