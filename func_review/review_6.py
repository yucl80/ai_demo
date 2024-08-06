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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 设置API调用限制（每分钟20次）
CALLS = 20
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_openai_api(messages):
    """
    调用OpenAI API，包含速率限制
    """
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

def get_code_diff(repo_path, commit_sha):
    """
    获取指定commit的代码变更
    """
    try:
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_sha)
        return commit.diff(commit.parents[0])
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"Invalid git repository: {repo_path}")
        sys.exit(1)
    except git.exc.GitCommandError as e:
        logging.error(f"Git command error: {str(e)}")
        sys.exit(1)

def review_code_change(file_path, old_content, new_content):
    """
    使用LLM review代码变更
    """
    prompt = f"""
    请review以下代码变更并提供意见：

    文件: {file_path}

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
    7. 总体评分（1-10）

    JSON格式示例：
    {{
        "code_quality": "评论...",
        "potential_bugs": "评论...",
        "performance_issues": "评论...",
        "readability_maintainability": "评论...",
        "best_practices": "评论...",
        "security_considerations": "评论...",
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
    file_path = diff.a_path if diff.a_path else diff.b_path
    old_content = diff.a_blob.data_stream.read().decode('utf-8') if diff.a_blob else "新文件"
    new_content = diff.b_blob.data_stream.read().decode('utf-8') if diff.b_blob else "文件已删除"

    logging.info(f"Reviewing changes in file: {file_path}")

    try:
        review_result = review_code_change(file_path, old_content, new_content)
        return file_path, review_result
    except Exception as e:
        logging.error(f"Error reviewing file {file_path}: {str(e)}")
        return file_path, None

def main(repo_path, commit_sha, max_workers=5):
    logging.info(f"Starting code review for commit {commit_sha} in repository {repo_path}")

    diffs = get_code_diff(repo_path, commit_sha)

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
                if key != 'overall_score':
                    print(f"- {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"\n### File: {file_path}")
            print("Review failed for this file.")

    # 保存结果到JSON文件
    with open('review_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("Code review completed. Results saved to review_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated code review using LLM")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("commit_sha", help="SHA of the commit to review")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker threads")
    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        logging.error(f"Repository path does not exist: {args.repo_path}")
        sys.exit(1)

    main(args.repo_path, args.commit_sha, args.max_workers)
