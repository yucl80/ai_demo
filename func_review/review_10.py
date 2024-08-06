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
from jira import JIRA
from gensim.summarization import summarize

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置API调用限制（每分钟20次）
CALLS = 20
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_openai_api(messages):
    """调用OpenAI API，包含速率限制"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content
    except openai.error.RateLimitError:
        logging.warning("OpenAI API rate limit reached. Waiting before retrying...")
        raise
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        raise

def run_static_analysis(file_path: str) -> Dict[str, Any]:
    """运行静态代码分析工具"""
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

def get_function_calls(node):
    """递归获取函数内的所有函数调用"""
    calls = []
    for child in node.get_children():
        if isinstance(child, astroid.Call):
            if isinstance(child.func, astroid.Name):
                calls.append(child.func.name)
            elif isinstance(child.func, astroid.Attribute):
                calls.append(child.func.attrname)
        calls.extend(get_function_calls(child))
    return calls

def build_call_graph(file_content):
    """构建函数调用图"""
    tree = astroid.parse(file_content)
    graph = nx.DiGraph()
    
    for node in tree.body:
        if isinstance(node, astroid.FunctionDef):
            graph.add_node(node.name)
            calls = get_function_calls(node)
            for call in calls:
                graph.add_edge(node.name, call)
    
    return graph

def analyze_imports(node: astroid.Module) -> List[str]:
    """分析模块的导入语句"""
    imports = []
    for child in node.body:
        if isinstance(child, (astroid.Import, astroid.ImportFrom)):
            for name in child.names:
                imports.append(name[0])
    return imports

def analyze_variable_scope(node: astroid.Module) -> Dict[str, List[str]]:
    """分析变量作用域"""
    scopes = {'global': [], 'local': []}
    for child in node.body:
        if isinstance(child, astroid.Assign):
            for target in child.targets:
                if isinstance(target, astroid.Name):
                    scopes['global'].append(target.name)
        elif isinstance(child, astroid.FunctionDef):
            scopes['local'].extend([arg.name for arg in child.args.args])
    return scopes

def analyze_docstrings(node: astroid.Module) -> Dict[str, str]:
    """分析文档字符串"""
    docstrings = {}
    for child in node.body:
        if isinstance(child, (astroid.FunctionDef, astroid.ClassDef)):
            docstrings[child.name] = ast.get_docstring(child) or "No docstring"
    return docstrings

def analyze_naming_conventions(node: astroid.Module) -> Dict[str, List[str]]:
    """分析命名约定"""
    conventions = {'snake_case': [], 'camelCase': [], 'PascalCase': []}
    for child in node.body:
        if isinstance(child, (astroid.FunctionDef, astroid.ClassDef)):
            if re.match(r'^[a-z_]+$', child.name):
                conventions['snake_case'].append(child.name)
            elif re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', child.name):
                conventions['camelCase'].append(child.name)
            elif re.match(r'^[A-Z][a-zA-Z]*$', child.name):
                conventions['PascalCase'].append(child.name)
    return conventions

def get_git_history(repo_path: str, file_path: str) -> List[Dict[str, Any]]:
    """获取文件的Git历史"""
    repo = git.Repo(repo_path)
    commits = list(repo.iter_commits(paths=file_path))
    return [{'sha': commit.hexsha, 'message': commit.message, 'author': commit.author.name, 'date': commit.committed_datetime} for commit in commits[:10]]  # 只获取最近10次提交

def get_jira_issues(jira_url: str, jira_token: str, project_key: str) -> List[Dict[str, Any]]:
    """获取JIRA问题"""
    jira = JIRA(server=jira_url, token_auth=jira_token)
    issues = jira.search_issues(f'project={project_key} AND status in (Open, "In Progress", Reopened)')
    return [{'key': issue.key, 'summary': issue.fields.summary, 'description': issue.fields.description} for issue in issues]

def get_function_class_context(file_content, start_line, end_line, repo_path, file_path):
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
        git_history = get_git_history(repo_path, file_path)
        
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
            'overall_complexity': sum(complexity.values()) / len(complexity) if complexity else 0,
            'git_history': git_history
        }
    except Exception as e:
        logging.error(f"Error parsing file for context: {str(e)}")
        return {}

def review_code_change(file_path: str, old_content: str, new_content: str, context: Dict[str, Any], static_analysis_result: Dict[str, Any], jira_issues: List[Dict[str, Any]], coding_standards: str):
    """使用LLM review代码变更，包含更多上下文信息，并提供风险评估"""
    context_str = json.dumps(context, indent=2)
    static_analysis_str = json.dumps(static_analysis_result, indent=2)
    jira_issues_str = json.dumps(jira_issues, indent=2)
    
    prompt = f"""
    请review以下代码变更并提供意见，包括修改建议的风险等级：

    文件: {file_path}

    代码上下文和分析结果:
    {context_str}

    静态分析结果:
    {static_analysis_str}

    相关JIRA问题:
    {jira_issues_str}

    项目编码标准:
    {coding_standards}

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
    13. 与项目编码标准的符合度
    14. 与相关JIRA问题的关联性
    15. 总体评分（1-10）
    16. 修改建议及其风险等级（低、中、高）

    对于每个修改建议，请提供以下信息：
    - 建议内容
    - 风险等级（低、中、高）
    - 风险评估理由
    - 预期影响

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
        "coding_standards_compliance": "评论...",
        "jira_issues_relevance": "评论...",
        "overall_score": 8,
        "modification_suggestions": [
            {{
                "suggestion": "建议内容...",
                "risk_level": "中",
                "risk_assessment": "评估理由...",
                "expected_impact": "预期影响..."
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

def get_code_diff(repo_path, commit_sha, jira_url, jira_token, project_key, coding_standards_path):
    """获取指定commit的代码变更，并包含上下文信息和静态分析结果"""
    try:
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_sha)
        diffs = commit.diff(commit.parents[0])
        
        # 读取项目编码标准
        with open(coding_standards_path, 'r') as f:
            coding_standards = f.read()

        # 获取JIRA问题
        jira_issues = get_jira_issues(jira_url, jira_token, project_key)
        
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
            context = get_function_class_context(new_content, start_line, end_line, repo_path, file_path)
            
            # 运行静态分析
            static_analysis_result = run_static_analysis(file_path)
            
            enhanced_diffs.append({
                'file_path': file_path,
                'old_content': old_content,
                'new_content': new_content,
                'context': context,
                'static_analysis': static_analysis_result,
                'jira_issues': jira_issues,
                'coding_standards': coding_standards
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
    jira_issues = diff['jira_issues']
    coding_standards = diff['coding_standards']

    logging.info(f"Reviewing changes in file: {file_path}")

    try:
        review_result = review_code_change(file_path, old_content, new_content, context, static_analysis, jira_issues, coding_standards)
        return file_path, review_result
    except Exception as e:
        logging.error(f"Error reviewing file {file_path}: {str(e)}")
        return file_path, None

def main(repo_path, commit_sha, jira_url, jira_token, project_key, coding_standards_path, max_workers=5):
    logging.info(f"Starting code review for commit {commit_sha} in repository {repo_path}")

    diffs = get_code_diff(repo_path, commit_sha, jira_url, jira_token, project_key, coding_standards_path)

    results = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(review_file, diff): diff for diff in diffs}
        for future in as_completed(future_to_file):
            file_path, review_result = future.result()
            results[file_path] = review_result

    end_time = time.time()
    total_time = end_time - start_time

    print("\n## Code Review Summary")
    print(f"Total files reviewed: {len(results)}")
    print(f"Total time taken: {total_time:.2f} seconds")

    overall_scores = [result['overall_score'] for result in results.values() if result and 'overall_score' in result]
    if overall_scores:
        average_score = sum(overall_scores) / len(overall_scores)
        print(f"Average overall score: {average_score:.2f}")

    for file_path, review in results.items():
        if review:
            print(f"\n### File: {file_path}")
            print(f"Overall Score: {review['overall_score']}")
            print("\nKey Points:")
            for key, value in review.items():
                if key != 'overall_score' and key != 'modification_suggestions':
                    print(f"- {key.replace('_', ' ').title()}: {value}")
            
            print("\nModification Suggestions:")
            for suggestion in review['modification_suggestions']:
                print(f"- Suggestion: {suggestion['suggestion']}")
                print(f"  Risk Level: {suggestion['risk_level']}")
                print(f"  Risk Assessment: {suggestion['risk_assessment']}")
                print(f"  Expected Impact: {suggestion['expected_impact']}")
        else:
            print(f"\n### File: {file_path}")
            print("Review failed for this file.")

    # 保存结果到JSON文件
    with open('review_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("Code review completed. Results saved to review_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated code review using LLM with enhanced context analysis and risk assessment")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("commit_sha", help="SHA of the commit to review")
    parser.add_argument("jira_url", help="JIRA server URL")
    parser.add_argument("jira_token", help="JIRA API token")
    parser.add_argument("project_key", help="JIRA project key")
    parser.add_argument("coding_standards_path", help="Path to the coding standards file")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker threads")
    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        logging.error(f"Repository path does not exist: {args.repo_path}")
        sys.exit(1)

    main(args.repo_path, args.commit_sha, args.jira_url, args.jira_token, args.project_key, args.coding_standards_path, args.max_workers)
