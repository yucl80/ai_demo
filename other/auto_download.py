from huggingface_hub import hf_hub_download
import time

def download_with_retry(repo_id, filename, max_retries=3):
    retries = 0
    while retries <= max_retries:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"Successfully downloaded {filename} from {repo_id}.")
            return path
        except Exception as e:
            print(f"Download failed with error: {e}. Retrying in 5 seconds...")
            retries += 1
            time.sleep(5)
    print("Failed to download after multiple retries.")
    return None

# 使用示例
model_id = "meetkai/functionary-small-v2.5"
filenames = ["model-00001-of-00004.safetensors","model-00002-of-00004.safetensors","model-00003-of-00004.safetensors","model-00004-of-00004.safetensors"]
for file in filenames:
  download_with_retry(model_id, file,10000)
  
  
import requests
from requests.exceptions import RequestException
import time

def download_file_with_retry(url, retries=3, delay=5):
    for i in range(retries):
        try:
            response = requests.get(url, stream=True)
            # 检查请求是否成功
            response.raise_for_status()  # 如果响应状态码不是200，将抛出HTTPError异常
            with open("filename.ext", "wb") as f:  # 请根据实际情况替换"filename.ext"
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉keep-alive新行
                        f.write(chunk)
            print("下载完成。")
            return True
        except RequestException as e:
            print(f"下载失败，原因：{e}. 尝试重新下载 ({i+1}/{retries})...")
            time.sleep(delay)  # 等待一段时间后重试
    print("下载尝试达到最大次数，仍未成功。")
    return False

url = "http://example.com/path/to/your/file.ext"  # 请替换为实际的URL
download_file_with_retry(url) 