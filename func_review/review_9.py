import os
import sys
import openai
import git
from dotenv import load_dotenv
import argparse
import logging
from ratelimit import limits, sleep_and_retry
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast
import astroid
import networkx as nx
import re
from typing import List, Dict, Any
import subprocess

# ... [之前的导入和设置保持不变] ...

def run_static_analysis(file_path: str) -> Dict[str, Any]:
    """运行静态代码分析工具"""
    # 这里使用pylint作为示例,你可以根据需要替换或添加其他工具
    try:
        result = subprocess.run(['pylint', file_path, '--output-format=json'], capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        logging.error(f"Error running static analysis: {str(e)}")
        return {}

def calculate_cyclomatic_complexity(node: astroid.FunctionDef) -> int:
    """计算函数的圈复杂度"""
    complexity = 1
    for child in node.get_children():
        if isinstance(child, (astroid.If, astroid.While, astroid.For, astroid.Assert)):
            complexity += 1
        elif isinstance(child, astroid.BoolOp) and child.op == 'or':
            complexity += len(child.values) - 1
    return complexity

def analyze_code_complexity(file_content: str) -> Dict[str, int]:
    """分析代码复杂度"""
    tree = astroid.parse(file_content)
    complexity = {}
    for node in tree.body:
        if isinstance(node, astroid.FunctionDef):
            complexity[node.name] = calculate_cyclomatic_complexity(node)
    return complexity

def get_function_class_context(file_content, start_line, end_line):
    """获取指定行范围内的函数或类定义的上下文，包括函数调用关系和其他分析"""
    try:
        tree = astroid.parse(file_content)
        context = []
        call_graph = build_call_graph(file_content)
        imports = analyze_imports(tree)
        variable_scopes = analyze_variable_scope(tree)
        docstrings = analyze_docstrings(tree)
        naming_conventions = analyze_naming_conventions(tree)
        complexity = analyze_code_complexity(file_content)
        
        for node in tree.body:
            if isinstance(node, (astroid.FunctionDef, astroid.ClassDef)):
                if node.lineno <= end_line and node.end_lineno >= start_line:
                    context_item = {
                        'type': 'function' if isinstance(node, astroid.FunctionDef) else 'class',
                        'name': node.name,
                        'start': node.lineno,
                        'end': node.end_lineno,
                        'docstring': docstrings.get(node.name, "No docstring"),
                        'complexity': complexity.get(node.name, 0) if isinstance(node, astroid.FunctionDef) else None
                    }
                    
                    if isinstance(node, astroid.FunctionDef):
                        context_item['calls'] = list(call_graph.successors(node.name))
                        context_item['called_by'] = list(call_graph.predecessors(node.name))
                        context_item['local_variables'] = variable_scopes['local']
                    
                    context.append(context_item)
        
        return {
            'context': context,
            'imports': imports,
            'global_variables': variable_scopes['global'],
            'naming_conventions': naming_conventions,
            'overall_complexity': sum(complexity.values()) / len(complexity) if complexity else 0
        }
    except Exception as e:
        logging.error(f"Error parsing file for context: {str(e)}")
        return {}

def review_code_change(file_path: str, old_content: str, new_content: str, context: Dict[str, Any], static_analysis_result: Dict[str, Any]):
    """使用LLM review代码变更，包含更多上下文信息，并提供风险评估"""
    context_str = json.dumps(context, indent=2)
    static_analysis_str = json.dumps(static_analysis_result, indent=2)
    prompt = f"""
    请review以下代码变更并提供意见，包括修改建议的风险等级：

    文件: {file_path}

    代码上下文和分析结果:
    {context_str}

    静态分析结果:
    {static_analysis_str}

    旧代码:
    {old_content}

    新代码:
    {new_content}

    请提供以下方面的review意见，并以JSON格式返回结果：
    1. 代码质量
    2. 潜在的bug
    3. 性能问题
    4. 可读性和可维护性
    5. 最佳实践遵循情况
    6. 安全性考虑
    7. 与上下文的一致性和影响
    8. 函数调用关系的变化及其影响
    9. 导入和依赖关系的变化
    10. 变量作用域和命名约定的遵循情况
    11. 文档和注释的完整性和准确性
    12. 代码复杂度变化
    13. 总体评分（1-10）
    14. 修改建议及其风险等级（低、中、高）

    对于每个修改建议，请提供以下信息：
    - 建议内容
    - 风险等级（低、中、高）
    - 风险评估理由

    JSON格式示例：
    {{
        "code_quality": "评论...",
        "potential_bugs": "评论...",
        "performance_issues": "评论...",
        "readability_maintainability": "评论...",
        "best_practices": "评论...",
        "security_considerations": "评论...",
        "context_consistency": "评论...",
        "function_call_impact": "评论...",
        "import_dependency_changes": "评论...",
        "variable_scope_naming": "评论...",
        "documentation_comments": "评论...",
        "complexity_changes": "评论...",
        "overall_score": 8,
        "modification_suggestions": [
            {{
                "suggestion": "建议内容...",
                "risk_level": "中",
                "risk_assessment": "评估理由..."
            }},
            ...
        ]
    }}
    """

    messages = [
        {"role": "system", "content": "You are a senior software engineer performing a code review. Provide your review in JSON format."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai_api(messages)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON response for file {file_path}")
        return None

def get_code_diff(repo_path, commit_sha):
    """获取指定commit的代码变更，并包含上下文信息和静态分析结果"""
    try:
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_sha)
        diffs = commit.diff(commit.parents[0])
        
        enhanced_diffs = []
        for diff in diffs:
            file_path = diff.a_path if diff.a_path else diff.b_path
            old_content = diff.a_blob.data_stream.read().decode('utf-8') if diff.a_blob else ""
            new_content = diff.b_blob.data_stream.read().decode('utf-8') if diff.b_blob else ""
            
            # 获取变更的行号范围
            diff_lines = list(diff.diff.decode('utf-8').split('\n'))
            changed_lines = [line for line in diff_lines if line.startswith('+') or line.startswith('-')]
            start_line = diff_lines.index(changed_lines[0]) if changed_lines else 0
            end_line = diff_lines.index(changed_lines[-1]) if changed_lines else len(diff_lines)
            
            # 获取上下文
            context = get_function_class_context(new_content, start_line, end_line)
            
            # 运行静态分析
            static_analysis_result = run_static_analysis(file_path)
            
            enhanced_diffs.append({
                'file_path': file_path,
                'old_content': old_content,
                'new_content': new_content,
                'context': context,
                'static_analysis': static_analysis_result
            })
        
        return enhanced_diffs
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"Invalid git repository: {repo_path}")
        sys.exit(1)
    except git.exc.GitCommandError as e:
        logging.error(f"Git command error: {str(e)}")
        sys.exit(1)

def review_file(diff):
    file_path = diff['file_path']
    old_content = diff['old_content']
    new_content = diff['new_content']
    context = diff['context']
    static_analysis = diff['static_analysis']

    logging.info(f"Reviewing changes in file: {file_path}")

    try:
        review_result = review_code_change(file_path, old_content, new_content, context, static_analysis)
        return file_path, review_result
    except Exception as e:
        logging.error(f"Error reviewing file {file_path}: {str(e)}")
        return file_path, None

# ... [main函数和其他部分保持不变] ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated code review using LLM with enhanced context analysis and risk assessment")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("commit_sha", help="SHA of the commit to review")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker threads")
    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        logging.error(f"Repository path does not exist: {args.repo_path}")
        sys.exit(1)

    main(args.repo_path, args.commit_sha, args.max_workers)
