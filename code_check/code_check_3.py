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
import ast
import math

# ... [前面的代码保持不变] ...

class CodeStyleChecker:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.rules = self.load_rules(self.config['rules_file'])
        openai.api_key = self.config['api_key']
        self.max_rules_per_check = self.config['max_rules_per_check']
        self.cache = {}
        self.base_threshold = self.config.get('base_threshold', 0.7)

    # ... [其他方法保持不变] ...

    def check_code(self, code: str, file_path: str, category: str = None, model: str = "gpt-3.5-turbo") -> Dict:
        code_hash = self.get_code_hash(code)
        if code_hash in self.cache:
            return self.cache[code_hash]

        if category and category in self.rules:
            rules = self.rules[category]
        else:
            rules = [rule for category in self.rules.values() for rule in category]

        # 按权重排序规则
        rules.sort(key=lambda x: x['weight'], reverse=True)

        # 获取代码上下文信息
        context_info = self.get_code_context(code, file_path)

        # 计算动态阈值
        dynamic_threshold = self.calculate_dynamic_threshold(context_info)

        results = []
        for i in range(0, len(rules), self.max_rules_per_check):
            chunk = rules[i:i + self.max_rules_per_check]
            prompt = self.create_prompt(code, chunk, context_info, dynamic_threshold)
            try:
                response = self.call_openai_with_retry(model, prompt)
                results.append(response)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI API error: {str(e)}")
                results.append(f"Error: {str(e)}")

        final_result = "\n\n".join(results)
        analysis_result = {
            'feedback': final_result,
            'threshold': dynamic_threshold,
            'context': context_info
        }
        self.cache[code_hash] = analysis_result
        return analysis_result

    def calculate_dynamic_threshold(self, context_info: Dict) -> float:
        # 基于代码复杂性和重要性计算动态阈值
        complexity_factor = self.calculate_complexity_factor(context_info)
        importance_factor = self.calculate_importance_factor(context_info)
        
        # 使用基础阈值和因子来计算动态阈值
        dynamic_threshold = self.base_threshold * complexity_factor * importance_factor
        
        # 确保阈值在合理范围内
        return max(0.5, min(0.95, dynamic_threshold))

    def calculate_complexity_factor(self, context_info: Dict) -> float:
        # 基于代码度量计算复杂性因子
        metrics = context_info['code_metrics']
        ast_info = context_info['ast_info']
        
        # 考虑代码行数、函数数量、类数量等
        total_lines = metrics['total_lines']
        num_functions = len(ast_info['functions'])
        num_classes = len(ast_info['classes'])
        
        # 简单的复杂性计算公式，可以根据需要调整
        complexity = (total_lines / 100) + (num_functions / 5) + (num_classes / 2)
        return 1 + (math.log(complexity + 1) / 10)  # 使用对数避免因子增长过快

    def calculate_importance_factor(self, context_info: Dict) -> float:
        # 基于文件名和目录计算重要性因子
        file_name = context_info['file_name'].lower()
        directory = context_info['directory'].lower()
        
        importance = 1.0
        
        # 检查是否是核心模块或关键文件
        if 'core' in directory or 'main' in file_name:
            importance *= 1.2
        
        # 检查是否是测试文件
        if 'test' in file_name or 'test' in directory:
            importance *= 0.9
        
        # 其他重要性判断逻辑...
        
        return importance

    def create_prompt(self, code: str, rules: List[Dict], context_info: Dict, threshold: float) -> str:
        prompt = f"Please analyze the following code and check if it complies with these rules. Use a threshold of {threshold:.2f} for rule violations:\n\n"
        # ... [其余的提示创建逻辑保持不变] ...
        return prompt

    # ... [其他方法保持不变] ...

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
                .threshold { color: #888; }
            </style>
        </head>
        <body>
            <h1>Code Style Check Results</h1>
            {% for filename, result in results.items() %}
            <div class="file">
                <div class="filename">{{ filename }}</div>
                <div class="threshold">Threshold: {{ result.threshold|round(2) }}</div>
                <div class="feedback">{{ result.feedback }}</div>
            </div>
            {% endfor %}
        </body>
        </html>
        """)
        return template.render(results=results)

# ... [main函数和其他部分保持不变] ...
