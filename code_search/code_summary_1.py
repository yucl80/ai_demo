import ast
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 初始化 LLM 模型
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

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

def generate_code_summary(code, call_graph):
    # 提取函数调用关系
    call_graph_info = "\n".join([f"{caller} -> {callee}" for caller, callee in call_graph.edges])
    
    # 将调用关系与代码片段结合
    input_text = f"Code:\n{code}\n\nCall Graph:\n{call_graph_info}"
    
    # 使用 LLM 生成代码摘要
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

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
