import git
import openai
import json

# 设置OpenAI API密钥
openai.api_key = 'your_openai_api_key'

def get_latest_changes(repo_path):
    """获取最新的代码变更文件列表"""
    repo = git.Repo(repo_path)
    diff = repo.git.diff('HEAD~1', 'HEAD', '--name-only')
    changed_files = diff.split('\n')
    return changed_files

def get_file_diff(repo_path, file_path):
    """获取文件的具体改动内容"""
    repo = git.Repo(repo_path)
    diff = repo.git.diff('HEAD~1', 'HEAD', file_path)
    return diff

def summarize_changes(diff):
    """使用LLM总结代码变更点"""
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Summarize the following code changes:\n{diff}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def summarize_impact(diff):
    """使用LLM分析代码变更的关联影响"""
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Analyze the impact of the following code changes on the overall system:\n{diff}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def generate_report(summaries, output_path):
    """生成报告并保存为JSON文件"""
    report = [{'file': file, 'changes_summary': summary['changes_summary'], 'impact_summary': summary['impact_summary']} for file, summary in summaries.items()]
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

# 主流程
repo_path = '/path/to/your/repo'
changed_files = get_latest_changes(repo_path)
diffs = {file: get_file_diff(repo_path, file) for file in changed_files}

summaries = {}
for file, diff in diffs.items():
    changes_summary = summarize_changes(diff)
    impact_summary = summarize_impact(diff)
    summaries[file] = {'changes_summary': changes_summary, 'impact_summary': impact_summary}

report_path = 'code_review_report.json'
generate_report(summaries, report_path)
print(f"Report generated: {report_path}")

# 输出查看每个文件的变更总结和影响分析
for file, summary in summaries.items():
    print(f"Summary for {file}:\nChanges Summary:\n{summary['changes_summary']}\nImpact Summary:\n{summary['impact_summary']}\n")
