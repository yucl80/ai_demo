from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import ast
import networkx as nx
import matplotlib.pyplot as plt

def extract_code_from_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_function_calls_llm(code_content):
    """使用LLM分析代码中的函数调用关系"""
    # 初始化LLM
    llm = Ollama(model="mistral", request_timeout=60.0)
    
    # 构建提示词
    prompt = f"""分析以下Python代码，列出所有的函数调用关系。
    对于每个函数，说明它调用了哪些其他函数。
    只需要返回函数名和调用关系，格式如下：
    caller_function -> called_function
    
    代码内容:
    {code_content}
    """
    
    # 获取LLM的分析结果
    response = llm.complete(prompt)
    return response.text

def parse_llm_response(response):
    """解析LLM返回的结果，转换为图结构"""
    G = nx.DiGraph()
    
    # 按行分割响应
    lines = response.strip().split('\n')
    for line in lines:
        if '->' in line:
            caller, callee = map(str.strip, line.split('->'))
            G.add_edge(caller, callee)
    
    return G

def visualize_call_graph(G, output_file='function_calls.png'):
    """可视化函数调用图"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray', arrowsize=20)
    
    plt.title("Function Call Graph")
    plt.savefig(output_file)
    plt.close()

def analyze_code_with_llm(file_path):
    """主函数：分析代码并生成调用图"""
    # 读取代码
    code_content = extract_code_from_file(file_path)
    
    # 使用LLM分析
    llm_analysis = get_function_calls_llm(code_content)
    
    # 解析结果
    call_graph = parse_llm_response(llm_analysis)
    
    # 可视化
    visualize_call_graph(call_graph)
    
    return llm_analysis

if __name__ == "__main__":
    # 示例使用
    file_path = "d:/workspaces/python_projects/ai_demo/code_search/code_search_5.py"
    analysis_result = analyze_code_with_llm(file_path)
    print("Analysis result:")
    print(analysis_result)
