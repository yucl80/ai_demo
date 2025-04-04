import ast
from typing import List, Dict, Any
from collections import deque
from analyzers import CodeIndexer, BusinessLogicAnalyzer

class LargeCodeAnalyzer:
    def __init__(self, llm_api):
        self.llm_api = llm_api
        self.max_tokens = 7500
        self.context_snapshots = deque(maxlen=10)
        self.global_context = {}
        self.code_indexer = CodeIndexer()
        self.business_logic_analyzer = BusinessLogicAnalyzer(llm_api)

    async def analyze_code(self, code_files: List[str]) -> Dict[str, Any]:
        """
        Analyze code files and extract insights.
        """
        # Index all files for searching
        for file_path in code_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.code_indexer.add_to_index(file_path, content, "file")

        # Analyze business logic across files
        business_logic_analysis = await self.business_logic_analyzer.analyze_business_logic(code_files)
        
        # Store context snapshot
        self.context_snapshots.append({
            "files_analyzed": code_files,
            "business_logic": business_logic_analysis
        })

        return {
            "business_logic_analysis": business_logic_analysis,
            "indexed_files": list(self.code_indexer.index.keys())
        }

    def search_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Search through indexed code using the query.
        """
        return self.code_indexer.search(query)

    def load_context_snapshot(self, snapshot_index: int = -1) -> Dict[str, Any]:
        """
        Load a previous context snapshot.
        """
        if not self.context_snapshots:
            return {}
        return list(self.context_snapshots)[snapshot_index]

# Example usage
async def main():
    api_key = "your-api-key-here"
    from LLMApi_2 import EnhancedLLMApi
    
    llm_api = EnhancedLLMApi(api_key)
    analyzer = LargeCodeAnalyzer(llm_api)
    
    code_files = [
        "path/to/your/file1.py",
        "path/to/your/file2.py"
    ]
    
    analysis_result = await analyzer.analyze_code(code_files)
    print(analysis_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())