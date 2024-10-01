import torch
from transformers import RobertaTokenizer, T5EncoderModel

# 加载CodeT5模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5EncoderModel.from_pretrained('Salesforce/codet5-base')

# 示例代码片段
code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

# 使用分词器编码代码
inputs = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512)

# 获取编码后的向量表示
with torch.no_grad():
    outputs = model(**inputs)

# 提取代码向量
code_vector = outputs.last_hidden_state

# 对所有token取平均值作为整个代码片段的向量表示
code_embedding = torch.mean(code_vector, dim=1)
print("代码片段的向量表示: ", code_embedding)
