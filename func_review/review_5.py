import os
import git
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
# openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

def get_code_diff(repo_path, commit_sha):
    """
    获取指定commit的代码变更
    """
    repo = git.Repo(repo_path)
    commit = repo.commit(commit_sha)
    return commit.diff(commit.parents[0])

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

    请提供以下方面的review意见：
    1. 代码质量
    2. 潜在的bug
    3. 性能问题
    4. 可读性和可维护性
    5. 最佳实践遵循情况
    """


    response = client.chat.completions.create(
        model="gpt-4",  # 使用最新的GPT-4模型
        messages=[
            {"role": "system", "content": "You are a senior software engineer performing a code review."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def main():
    repo_path = "d:/workspaces/python_projects"  # 替换为你的git仓库路径
    commit_sha = "c8c9979"  # 替换为你要review的commit SHA

    diffs = get_code_diff(repo_path, commit_sha)

    for diff in diffs:
        file_path = diff.a_path if diff.a_path else diff.b_path
        old_content = diff.a_blob.data_stream.read().decode('utf-8') if diff.a_blob else "新文件"
        new_content = diff.b_blob.data_stream.read().decode('utf-8') if diff.b_blob else "文件已删除"

        review_comment = review_code_change(file_path, old_content, new_content)
        print(f"File: {file_path}")
        print("Review Comment:")
        print(review_comment)
        print("-" * 50)

if __name__ == "__main__":
    main()
