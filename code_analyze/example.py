import asyncio
from enhanced_llm_api import EnhancedLLMApi

async def main():
    # Initialize the enhanced LLM API with your API key
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    # Example code to analyze
    code_to_analyze = """
    def calculate_fibonacci(n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        if n <= 2:
            return 1
        return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
    """
    
    # Perform enhanced analysis
    analysis_result = await llm_api.enhanced_analysis(code_to_analyze)
    
    # Print results
    print("\nCode Summary:")
    print(analysis_result["code_summary"])
    
    print("\nExpert Analyses:")
    for expert_name, analysis in analysis_result["expert_analyses"].items():
        print(f"\n{expert_name.replace('_', ' ').title()}:")
        print(analysis)
    
    print("\nFinal Analysis:")
    print(analysis_result["enhanced_analysis"])

if __name__ == "__main__":
    asyncio.run(main())
