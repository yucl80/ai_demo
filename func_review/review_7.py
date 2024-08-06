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

# ... [之前的导入和设置保持不变] ...

def get_function_class_context(file_content, start_line, end_line):
    """
    获取指定行范围内的函数或类定义的上下文
    """
    try:
        tree = astroid.parse(file_content)
        context = []
        for node in tree.body:
            if isinstance(node, (astroid.FunctionDef, astroid.ClassDef)):
                if node.lineno <= end_line and node.end_lineno >= start_line:
                    context.append({
                        'type': 'function' if isinstance(node, astroid.FunctionDef) else 'class',
                        'name': node.name,
                        'start': node.lineno,
                        'end': node.end_lineno,
                        'docstring': ast.get_docstring(node) or "No docstring"
                    })
        return context
    except Exception as e:
        logging.error(f"Error parsing file for context: {str(e)}")
        return []

def get_code_diff(repo_path, commit_sha):
    """
    获取指定commit的代码变更，并包含上下文信息
    """
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
            
            enhanced_diffs.append({
                'file_path': file_path,
                'old_content': old_content,
                'new_content': new_content,
                'context': context
            })
        
        return enhanced_diffs
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"Invalid git repository: {repo_path}")
        sys.exit(1)
    except git.exc.GitCommandError as e:
        logging.error(f"Git command error: {str(e)}")
        sys.exit(1)

def review_code_change(file_path, old_content, new_content, context):
    """
    使用LLM review代码变更，包含上下文信息
    """
    context_str = json.dumps(context, indent=2)
    prompt = f"""
    请review以下代码变更并提供意见：

    文件: {file_path}

    代码上下文:
    {context_str}

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
    8. 总体评分（1-10）

    JSON格式示例：
    {{
        "code_quality": "评论...",
        "potential_bugs": "评论...",
        "performance_issues": "评论...",
        "readability_maintainability": "评论...",
        "best_practices": "评论...",
        "security_considerations": "评论...",
        "context_consistency": "评论...",
        "overall_score": 8
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

def review_file(diff):
    file_path = diff['file_path']
    old_content = diff['old_content']
    new_content = diff['new_content']
    context = diff['context']

    logging.info(f"Reviewing changes in file: {file_path}")

    try:
        review_result = review_code_change(file_path, old_content, new_content, context)
        return file_path, review_result
    except Exception as e:
        logging.error(f"Error reviewing file {file_path}: {str(e)}")
        return file_path, None

# ... [main函数和其他部分保持不变] ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated code review using LLM with context analysis")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("commit_sha", help="SHA of the commit to review")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker threads")
    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        logging.error(f"Repository path does not exist: {args.repo_path}")
        sys.exit(1)

    main(args.repo_path, args.commit_sha, args.max_workers)
