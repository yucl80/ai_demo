from transformers import RobertaTokenizer, RobertaModel
import torch

# 加载GraphCodeBERT模型
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
model = RobertaModel.from_pretrained('microsoft/graphcodebert-base')

# 定义Python和Java代码片段
python_code = """
def fetch_data():
    import requests
    response = requests.get('http://example.com/api/data')
    return response.json()
"""

java_code = """
public class DataFetcher {
    public String fetchData() {
        HttpURLConnection conn = (HttpURLConnection) new URL("http://example.com/api/data").openConnection();
        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        return reader.readLine();
    }
}
"""

# Tokenize代码
inputs_python = tokenizer(python_code, return_tensors='pt', truncation=True, padding=True)
inputs_java = tokenizer(java_code, return_tensors='pt', truncation=True, padding=True)

# 提取代码的语义特征
with torch.no_grad():
    python_embedding = model(**inputs_python).last_hidden_state[:, 0, :]
    java_embedding = model(**inputs_java).last_hidden_state[:, 0, :]

print(len(python_embedding[0]))
print(len(java_embedding))

def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state[:, 0, :]
    return embedding

# 计算相似度（跨语言代码片段的依赖关系）
similarity = torch.cosine_similarity(python_embedding, java_embedding)
print(f'跨语言代码片段相似度：{similarity.item()}')

code_list = [
    'return maximum value',
    'def f(a,b): if a>b: return a else return b',
    'def f(x,y): if x>y: return y else return x'
]


a = get_code_embedding(code_list[0])
b = get_code_embedding(code_list[1])
c = get_code_embedding(code_list[2])


from sentence_transformers.util import cos_sim
print(cos_sim(a, b))
print(cos_sim(a, c))
print(cos_sim(b, c))
