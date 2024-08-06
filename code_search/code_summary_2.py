import ast
import networkx as nx
from transformers import AutoTokenizer, AutoModel, RobertaForCausalLM
import torch
import openai

# 初始化 LLM 模型
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = RobertaForCausalLM.from_pretrained("microsoft/codebert-base")

from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

def code_summary(max_length,code_source):
    source = f"""
        {code_source}
        """
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-iMat",       
        messages=[
            {"role": "system", "content": "给下面代码生成摘要，长度不超过20个字："},
            {"role": "user", "content":  source},
        ],
        temperature=0.7,  
    )
    return completion.choices[0].message.content
    
    

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = RobertaForCausalLM.from_pretrained("facebook/bart-large-cnn")


from transformers import RobertaTokenizer, T5ForConditionalGeneration

# tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
# model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.call_graph = nx.DiGraph()
        self.current_function = None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.call_graph.add_node(node.name)
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                self.call_graph.add_edge(self.current_function, node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.call_graph.add_edge(self.current_function, node.func.attr)
        self.generic_visit(node)


def build_call_graph(code):
    tree = ast.parse(code)
    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    return visitor.call_graph


def generate_code_summary(code, call_graph, max_length=512):
    # 提取函数调用关系
    call_graph_info = "\n".join(
        [f"{caller} -> {callee}" for caller, callee in call_graph.edges]
    )

    # 将调用关系与代码片段结合
    input_text = f"Code:\n{code}\n\nCall Graph:\n{call_graph_info}"

    # 分块处理
    input_chunks = [
        input_text[i : i + max_length] for i in range(0, len(input_text), max_length)
    ]

    summaries = []
    for chunk in input_chunks:
        summary = code_summary(max_length, chunk)
        summaries.append(summary)

    # 合并摘要
    final_summary = " ".join(summaries)

    return final_summary


# def localLLL(max_length, chunk):
#     inputs = tokenizer(
#         chunk, return_tensors="pt", max_length=max_length, truncation=True
#     )
#     summary_ids = model.generate(
#         inputs.input_ids,
#         max_length=150,
#         min_length=30,
#         length_penalty=2.0,
#         num_beams=4,
#         early_stopping=True,
#     )
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary


# 示例代码
code = """
def foo():
    bar()
    baz()

def bar():
    print("bar")

def baz():
    print("baz")
"""

# 构建函数调用图
call_graph = build_call_graph(code)

# 生成代码摘要
summary = generate_code_summary(code, call_graph)
print("Code Summary:")
print(summary)
