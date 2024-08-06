import openai
import json
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
from itertools import islice
import hashlib
from functools import lru_cache
import yaml
import time
from jinja2 import Template
import git

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeStyleChecker:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.rules = self.load_rules(self.config['rules_file'])
        openai.api_key = self.config['api_key']
        self.max_rules_per_check = self.config['max_rules_per_check']
        self.cache = {}

    def load_config(self, config_file: str) -> Dict:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_rules(self, rules_file: str) -> Dict:
        try:
            with open(rules_file, 'r') as f:
                rules = json.load(f)
            # 添加权重到规则中
            for category in rules:
                for rule in rules[category]:
                    if isinstance(rule, str):
                        rules[category] = {'rule': rule, 'weight': 1}
                    elif isinstance(rule, dict) and 'weight' not in rule:
                        rule['weight'] = 1
            return rules
        except FileNotFoundError:
            logging.error(f"Rules file not found: {rules_file}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in rules file: {rules_file}")
            raise

    @lru_cache(maxsize=100)
    def get_code_hash(self, code: str) -> str:
        return hashlib.md5(code.encode()).hexdigest()

    def check_code(self, code: str, category: str = None, model: str = "gpt-3.5-turbo") -> str:
        code_hash = self.get_code_hash(code)
        if code_hash in self.cache:
            return self.cache[code_hash]

        if category and category in self.rules:
            rules = self.rules[category]
        else:
            rules = [rule for category in self.rules.values() for rule in category]

        # 按权重排序规则
        rules.sort(key=lambda x: x['weight'], reverse=True)

        results = []
        for i in range(0, len(rules), self.max_rules_per_check):
            chunk = rules[i:i + self.max_rules_per_check]
            prompt = self.create_prompt(code, chunk)
            try:
                response = self.call_openai_with_retry(model, prompt)
                results.append(response)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI API error: {str(e)}")
                results.append(f"Error: {str(e)}")

        final_result = "\n\n".join(results)
        self.cache[code_hash] = final_result
        return final_result

    def create_prompt(self, code: str, rules: List[Dict]) -> str:
        prompt = f"Please analyze the following code and check if it complies with these rules:\n\n"
        for rule in rules:
            prompt += f"- {rule['rule']} (Weight: {rule['weight']})\n"
        prompt += f"\nCode to analyze:\n```\n{code}\n```\n"
        prompt += "Provide a detailed analysis, pointing out any violations and suggesting improvements."
        return prompt

    def call_openai_with_retry(self, model: str, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a code style checker. Analyze the given code and provide feedback based on the specified rules."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message['content']
            except openai.error.OpenAIError as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def batch_check(self, directory: str, output_file: str, category: str = None, model: str = "gpt-3.5-turbo"):
        results = {}
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            future_to_file = {executor.submit(self.check_file, os.path.join(directory, filename), category, model): filename
                              for filename in os.listdir(directory) if filename.endswith('.py')}
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    results[filename] = future.result()
                except Exception as exc:
                    logging.error(f'{filename} generated an exception: {exc}')

        self.save_results(results, output_file)

    def check_file(self, file_path: str, category: str = None, model: str = "gpt-3.5-turbo") -> str:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            return self.check_code(code, category, model)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return f"Error: {str(e)}"

    def save_results(self, results: Dict, output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        html_output = self.generate_html_report(results)
        html_file = output_file.rsplit('.', 1)[0] + '.html'
        with open(html_file, 'w') as f:
            f.write(html_output)

        logging.info(f"Results saved to {output_file} and {html_file}")

    def generate_html_report(self, results: Dict) -> str:
        template = Template("""
        <html>
        <head>
            <title>Code Style Check Results</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .file { margin-bottom: 20px; }
                .filename { font-weight: bold; }
                .feedback { white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <h1>Code Style Check Results</h1>
            {% for filename, feedback in results.items() %}
            <div class="file">
                <div class="filename">{{ filename }}</div>
                <div class="feedback">{{ feedback }}</div>
            </div>
            {% endfor %}
        </body>
        </html>
        """)
        return template.render(results=results)

    def incremental_check(self, repo_path: str, category: str = None, model: str = "gpt-3.5-turbo") -> Dict[str, str]:
        repo = git.Repo(repo_path)
        changed_files = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith('.py')]
        results = {}
        for file_path in changed_files:
            full_path = os.path.join(repo_path, file_path)
            results[file_path] = self.check_file(full_path, category, model)
        return results

def main():
    parser = argparse.ArgumentParser(description="Code Style Checker using LLM")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--directory", help="Directory containing Python files to check")
    parser.add_argument("--output", default="results.json", help="Output file for batch processing results")
    parser.add_argument("--category", help="Specific rule category to check")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--incremental", action="store_true", help="Perform incremental check on git repository")
    args = parser.parse_args()

    checker = CodeStyleChecker(args.config)

    if args.incremental:
        results = checker.incremental_check(os.getcwd(), args.category, args.model)
        checker.save_results(results, args.output)
    elif args.directory:
        checker.batch_check(args.directory, args.output, args.category, args.model)
    else:
        # 示例代码
        code_to_check = """
def calculate_sum(a,b):
    return a+b

def print_result(x):
    print('The result is: '+str(x))

result = calculate_sum(5, 10)
print_result(result)
        """

        feedback = checker.check_code(code_to_check, args.category, args.model)
        print("Feedback:")
        print(feedback)

if __name__ == "__main__":
    main()
