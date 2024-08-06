import asyncio
import time
import json
import argparse
from typing import List

async def main():
    parser = argparse.ArgumentParser(description="Analyze source code using LLM")
    parser.add_argument("--path", required=True, help="Path to the source code directory")
    parser.add_argument("--api-key", required=True, help="API key for the LLM service")
    parser.add_argument("--api-url", default="https://api.example.com", help="Base URL for the LLM API")
    parser.add_argument("--db-path", default="code_context.db", help="Path to the SQLite database")
    parser.add_argument("--git-repo", help="Path to the Git repository (optional)")
    args = parser.parse_args()

    try:
        start_time = time.time()
        llm_api = LLMApi(args.api_key, args.api_url)
        external_storage = ExternalStorage(args.db_path)
        analyzer = CodeAnalyzer(llm_api, external_storage)

        logger.info(f"Starting analysis of code in {args.path}")

        # 加载代码
        await analyzer.load_code(args.path)
        logger.info(f"Code loading completed in {time.time() - start_time:.2f} seconds")

        # 分析全局结构
        global_structure = await analyzer.analyze_global_structure()
        print("Global structure:")
        print(json.dumps(global_structure, indent=2))

        # 交互式分析
        while True:
            query = input("Enter a query about the code (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            answer = await analyzer.interactive_analysis(query)
            print("Answer:", answer)

        # 获取重构建议
        refactor_suggestions = await analyzer.refactor_suggestions()
        print("Refactoring suggestions:")
        for suggestion in refactor_suggestions:
            print(suggestion)

        # 检测代码克隆
        clones = await analyzer.detect_code_clones()
        print("Code clones detected:")
        for clone in clones:
            print(f"Files {clone[0]} and {clone[1]} are {clone[2]:.2f}% similar")

        # 生成文档
        documentation = await analyzer.generate_documentation()
        print("Generated documentation:")
        print(documentation)

        # 分批处理
        await analyzer.process_in_batches()

        # 获取上下文快照
        snapshot = analyzer.get_context_snapshot()
        print("Context snapshot created")

        # 使用多个模型进行分析
        multiple_models = [LLMApi(args.api_key, args.api_url) for _ in range(3)]  # 创建3个模型实例
        multi_model_analysis = await analyzer.analyze_with_multiple_models(multiple_models)
        print("Multi-model analysis results:")
        print(json.dumps(multi_model_analysis, indent=2))

        # 如果提供了Git仓库路径，分析代码演变
        if args.git_repo:
            evolution_analysis = await analyzer.analyze_code_evolution(args.git_repo)
            print("Code evolution analysis:")
            print(evolution_analysis)

        # 性能分析
        performance_metrics = analyzer.get_performance_metrics()
        print("Performance metrics:")
        print(json.dumps(performance_metrics, indent=2))

        # 安全漏洞检测
        vulnerabilities = await analyzer.detect_vulnerabilities()
        print("Detected vulnerabilities:")
        for vuln in vulnerabilities:
            print(f"- {vuln}")

        # 代码复杂度分析
        complexity_report = analyzer.analyze_complexity()
        print("Code complexity report:")
        print(json.dumps(complexity_report, indent=2))

        # 依赖关系可视化
        dependency_graph = analyzer.visualize_dependencies()
        print("Dependency graph generated and saved as 'dependency_graph.png'")

        logger.info(f"Total analysis time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())

