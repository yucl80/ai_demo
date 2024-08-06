import git
import openai
import json

# 设置OpenAI API密钥
openai.api_key = 'your_openai_api_key'

reviewPrompt = """`You are a senior developer tasked with reviewing the provided code patch. Your review should identify and categorize issues, highlighting potential bugs, suggesting performance optimizations, and flag security issues. Please be aware there maybe libraries or technologies present which you do not know. Format the review output as valid JSON. Each identified issue should be an object in an array, with each object including the following fields: 'category', 'description', 'suggestedCode', and 'codeSnippet'. The category should be one of 'Bugs', 'Performance', 'Security' or 'Style'. The suggestedCode should be an empty string if the recommendation is general or you do not have any code to fix the problem, otherwise return the suggested code to fix the problem. Make sure to escape any special characters in the suggestedCode and in the problematic codeSnippet. Output format: [{"category": "Bugs", "description": "<Describe the problem with the code>", "suggestedCode": "<Insert a code suggestion in the same language as the patch which fixes the issue>", "codeSnippet": "<Insert the problematic code from the patch>"}]. Return the array nothing else.`"""

# 获取最新的代码变更
def get_latest_changes(repo_path):
    repo = git.Repo(repo_path)
    diff = repo.git.diff('HEAD~1', 'HEAD', '--name-only')
    changed_files = diff.split('\n')
    return changed_files

# 获取文件改动的具体内容
def get_file_diff(repo_path, file_path):
    repo = git.Repo(repo_path)
    diff = repo.git.diff('HEAD~1', 'HEAD', file_path)
    return diff

# 使用LLM总结变更点
def summarize_changes(diff):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Summarize the following code changes:\n{diff}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# 使用LLM总结关联影响
def summarize_impact(diff):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Analyze the impact of the following code changes on the overall system:\n{diff}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# 生成简洁报告
def generate_report(summaries, output_path):
    report = [{'file': file, 'changes_summary': summary[0], 'impact_summary': summary[1]} for file, summary in summaries.items()]
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

# 主流程
repo_path = '/path/to/your/repo'
changed_files = get_latest_changes(repo_path)
diffs = {file: get_file_diff(repo_path, file) for file in changed_files}

summaries = {file: (summarize_changes(diff), summarize_impact(diff)) for file, diff in diffs.items()}

report_path = 'code_review_report.json'
generate_report(summaries, report_path)
print(f"Report generated: {report_path}")
