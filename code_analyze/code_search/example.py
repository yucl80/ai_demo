from search_engine import CodeSearchEngine
from code_analyzer import CodeAnalyzer, BusinessLogicSearcher
import logging
import json

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize components
    logger.info("Initializing search engine and analyzer...")
    search_engine = CodeSearchEngine()
    code_analyzer = CodeAnalyzer()
    logic_searcher = BusinessLogicSearcher(search_engine, code_analyzer)

    # Add repositories
    repos = [
        "d:/workspaces/python_projects/ai_demo/code_analyze",
        # Add more repository paths here
    ]

    for repo_path in repos:
        logger.info(f"Indexing repository: {repo_path}")
        search_engine.add_repository(repo_path)

    # Save the index
    logger.info("Saving search index...")
    search_engine.save_index("./search_index")

    # Example feature searches
    features = [
        "code search functionality",
        "analyze python code structure",
        "extract business logic from code"
    ]

    for feature in features:
        logger.info(f"\nSearching for feature: {feature}")
        results = logic_searcher.search_business_logic(feature)
        
        if not results:
            logger.info("No relevant business logic found.")
            continue
            
        for result in results:
            logger.info(f"\nFile: {result['file']} (score: {result['score']:.3f})")
            
            for element in result['elements']:
                if element['type'] == 'module':
                    logger.info(f"Module docstring: {element['docstring']}")
                    
                elif element['type'] == 'class':
                    logger.info(f"\nClass: {element['name']}")
                    if element['docstring']:
                        logger.info(f"Description: {element['docstring']}")
                    if element['base_classes']:
                        logger.info(f"Inherits from: {', '.join(element['base_classes'])}")
                        
                    for method in element['methods']:
                        logger.info(f"\n  Method: {method['name']}")
                        if method['docstring']:
                            logger.info(f"  Description: {method['docstring']}")
                        logger.info(f"  Parameters: {', '.join(method['params'])}")
                        logger.info(f"  Calls: {', '.join(method['calls'])}")
                        logger.info(f"  Complexity: {method['complexity']}")
                        logger.info(f"  Lines: {method['lines']}")
                        
                elif element['type'] == 'function':
                    logger.info(f"\nFunction: {element['name']}")
                    if element['docstring']:
                        logger.info(f"Description: {element['docstring']}")
                    logger.info(f"Parameters: {', '.join(element['params'])}")
                    logger.info(f"Calls: {', '.join(element['calls'])}")
                    logger.info(f"Complexity: {element['complexity']}")
                    logger.info(f"Lines: {element['lines']}")

if __name__ == "__main__":
    main()
