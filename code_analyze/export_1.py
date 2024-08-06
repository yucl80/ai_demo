from abc import ABC, abstractmethod
import asyncio
import json
from typing import Any, Dict



# 在EnhancedLLMApi类中更新方法


class ExpertModel(ABC):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        pass

class ArchitectureExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As an architecture expert, analyze the given code and context. Focus on:
        1. Overall code structure and design patterns
        2. Modularity and component interactions
        3. Scalability and maintainability of the architecture
        4. Suggestions for architectural improvements

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's architecture.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))

class PerformanceExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a performance expert, analyze the given code and context. Focus on:
        1. Algorithmic efficiency
        2. Resource usage (CPU, memory, I/O)
        3. Potential bottlenecks and performance hotspots
        4. Optimization suggestions

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's performance characteristics.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))

class SecurityExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a security expert, analyze the given code and context. Focus on:
        1. Potential security vulnerabilities
        2. Adherence to security best practices
        3. Data handling and privacy concerns
        4. Recommendations for security improvements

        Code:
        {code}

        Context:
        {context}

        Provide a detailed security analysis of the code.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))

class CodeQualityExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a code quality expert, analyze the given code and context. Focus on:
        1. Adherence to coding standards and best practices
        2. Code readability and maintainability
        3. Proper use of comments and documentation
        4. Suggestions for improving code quality

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's quality.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))

class TestingExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a testing expert, analyze the given code and context. Focus on:
        1. Test coverage and quality
        2. Potential edge cases and error scenarios
        3. Testability of the code
        4. Suggestions for improving test suite

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's testing aspects.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
    
class CodeSummaryExpert(ExpertModel):
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a code summarization expert, analyze the given code and context. Your task is to generate a high-level summary of the code that captures its main functionality, structure, and purpose. Focus on:

        1. The overall purpose of the code
        2. Key components or modules and their roles
        3. Main algorithms or processes implemented
        4. Important data structures used
        5. External dependencies and their purposes
        6. Any notable design patterns or architectural choices

        Provide a concise yet informative summary that would help a developer quickly understand the essence of this code without delving into every detail.

        Code:
        {code}

        Context:
        {context}

        Generate a comprehensive summary of the code in about 200-300 words.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))


class EnhancedLLMApi(LLMApi):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        self.experts = {
            "architecture": ArchitectureExpert(api_key),
            "performance": PerformanceExpert(api_key),
            "security": SecurityExpert(api_key),
            "code_quality": CodeQualityExpert(api_key),
            "testing": TestingExpert(api_key),
            "code_summary": CodeSummaryExpert(api_key)  # 添加新的专家模型
        }
        # ... (其他初始化代码保持不变)

    async def enhanced_analysis(self, code: str, repo_path: str = None) -> Dict[str, Any]:
        basic_analysis = await self.analyze_with_global_context(code)
        
        enhanced_context = {
            "ast_analysis": self.ast_analyzer.analyze(code),
            "git_analysis": self.git_analyzer.analyze(repo_path) if repo_path else None,
            "static_analysis": self.static_analyzer.analyze(code),
            "test_analysis": self.test_analyzer.analyze(repo_path) if repo_path else None,
            "dependency_analysis": self.dependency_analyzer.analyze(repo_path) if repo_path else None,
            "pattern_analysis": self.pattern_recognizer.recognize(code),
            "performance_analysis": self.performance_analyzer.analyze(code),
            "security_analysis": self.security_scanner.scan(code),
            "domain_insights": self.domain_knowledge.get_insights(code),
            "comments_analysis": self.comment_extractor.extract(code),
            "metaprogramming_analysis": self.metaprogramming_analyzer.analyze(code),
            "environment_analysis": self.environment_analyzer.analyze(repo_path) if repo_path else None
        }

        # 使用专家模型进行分析
        expert_analyses = await self._run_expert_analyses(code, enhanced_context)
        enhanced_context.update(expert_analyses)
        
        # 生成代码摘要
        code_summary = expert_analyses['code_summary_analysis']

        # 使用主模型整合所有分析结果
        final_analysis = await self._integrate_analyses(enhanced_context)

        visualization = self.generate_visualization(enhanced_context)

        return {
            **basic_analysis, 
            "enhanced_analysis": final_analysis,
            "expert_analyses": expert_analyses,
            "code_summary": code_summary,  # 添加代码摘要到返回结果
            "visualization": visualization
        }

    async def _run_expert_analyses(self, code: str, context: Dict[str, Any]) -> Dict[str, str]:
        expert_analyses = {}
        tasks = []
        for expert_name, expert in self.experts.items():
            tasks.append(self._run_expert_analysis(expert_name, expert, code, context))
        results = await asyncio.gather(*tasks)
        for expert_name, analysis in results:
            expert_analyses[f"{expert_name}_analysis"] = analysis
        return expert_analyses

    async def _run_expert_analysis(self, expert_name: str, expert: ExpertModel, code: str, context: Dict[str, Any]) -> Tuple[str, str]:
        analysis = await expert.analyze(code, context)
        return expert_name, analysis

    async def _integrate_analyses(self, context: Dict[str, Any]) -> str:
        integration_prompt = """
        As a senior software architect and code analyst, review and integrate the following analyses of a codebase:

        {context}

        Provide a comprehensive, high-level summary of the codebase, addressing:
        1. Overall architecture and design
        2. Code quality and maintainability
        3. Performance characteristics
        4. Security considerations
        5. Testing and reliability
        6. Areas for improvement and recommendations
        
        Also, consider the provided code summary and how it relates to the detailed analyses.

        Your summary should synthesize insights from all the expert analyses and provide a holistic view of the codebase.
        """
        return await self.analyze("", integration_prompt.format(context=json.dumps(context)))

    # ... (其他方法保持不变)
async def main():
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    code_to_analyze = """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    def load_data(file_path):
        return pd.read_csv(file_path)

    def preprocess_data(df):
        # Handle missing values
        df.dropna(inplace=True)
        # Convert categorical variables to numeric
        df = pd.get_dummies(df, drop_first=True)
        return df

    def split_data(X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def plot_results(y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

    def main():
        # Load and preprocess data
        df = load_data('data.csv')
        df = preprocess_data(df)

        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")

        # Plot results
        y_pred = model.predict(X_test)
        plot_results(y_test, y_pred)

    if __name__ == "__main__":
        main()
    """
    
    # 生成代码摘要
    summary = await llm_api.generate_code_summary(code_to_analyze)
    print("Code Summary:")
    print(summary)
    
    # 执行完整的增强分析
    analysis_result = await llm_api.enhanced_analysis(code_to_analyze)
    print("\nFull Analysis:")
    print(json.dumps(analysis_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
    
    
# 使用示例
async def main2():
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    large_file_path = "/path/to/your/large/file.py"
    analysis_result = await llm_api.analyze_large_file(large_file_path)
    
    print("File Summary:")
    print(analysis_result['summary'])
    
    print("\nBlock Analyses:")
    for block_analysis in analysis_result['block_analyses']:
        print(f"\n{block_analysis['type']} {block_analysis['name']}:")
        print(block_analysis['analysis'])
    
    print("\nFinal Integrated Analysis:")
    print(analysis_result['final_analysis'])
    
    print("\nGlobal Context:")
    print(analysis_result['global_context'])
    


