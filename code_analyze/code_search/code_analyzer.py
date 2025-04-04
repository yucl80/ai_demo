import ast
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import logging
import re

@dataclass
class FunctionInfo:
    name: str
    docstring: Optional[str]
    params: List[str]
    returns: List[str]
    calls: List[str]
    start_line: int
    end_line: int
    complexity: int
    
@dataclass
class ClassInfo:
    name: str
    docstring: Optional[str]
    methods: List[FunctionInfo]
    base_classes: List[str]
    start_line: int
    end_line: int

@dataclass
class ModuleInfo:
    file_path: str
    imports: List[str]
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    docstring: Optional[str]

class CodeAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: str) -> ModuleInfo:
        """Analyze a Python file and extract its structure and business logic."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            return self._analyze_module(tree, file_path)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return ModuleInfo(file_path, [], [], [], None)
    
    def _analyze_module(self, tree: ast.Module, file_path: str) -> ModuleInfo:
        """Extract information from a Python module."""
        imports = []
        classes = []
        functions = []
        docstring = ast.get_docstring(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
                    
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(self._analyze_class(node))
            elif isinstance(node, ast.FunctionDef):
                functions.append(self._analyze_function(node))
                
        return ModuleInfo(file_path, imports, classes, functions, docstring)
    
    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract information from a class definition."""
        methods = []
        base_classes = []
        
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(f"{base.value.id}.{base.attr}")
                
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(self._analyze_function(child))
                
        return ClassInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            methods=methods,
            base_classes=base_classes,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno
        )
    
    def _analyze_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """Extract information from a function definition."""
        calls = []
        returns = []
        complexity = 1  # Base complexity
        
        # Analyze function body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(f"{child.func.value.id}.{child.func.attr}")
            elif isinstance(child, ast.Return):
                if isinstance(child.value, ast.Name):
                    returns.append(child.value.id)
            # Calculate cyclomatic complexity
            elif isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
                
        # Get parameter names
        params = [arg.arg for arg in node.args.args]
        
        return FunctionInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            params=params,
            returns=returns,
            calls=list(set(calls)),  # Remove duplicates
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            complexity=complexity
        )
    
    def find_business_logic(self, module_info: ModuleInfo, feature_description: str) -> List[Dict]:
        """Find code elements related to a specific business feature."""
        results = []
        
        # Convert feature description to lowercase for case-insensitive matching
        feature_lower = feature_description.lower()
        
        # Helper function to check if text is related to feature
        def is_related(text: Optional[str]) -> bool:
            if not text:
                return False
            return any(word in text.lower() for word in feature_lower.split())
        
        # Check module-level docstring
        if is_related(module_info.docstring):
            results.append({
                'type': 'module',
                'file': module_info.file_path,
                'docstring': module_info.docstring
            })
        
        # Check classes
        for class_info in module_info.classes:
            class_related = False
            
            if is_related(class_info.docstring):
                class_related = True
            
            related_methods = []
            for method in class_info.methods:
                if is_related(method.docstring):
                    related_methods.append({
                        'name': method.name,
                        'docstring': method.docstring,
                        'params': method.params,
                        'calls': method.calls,
                        'complexity': method.complexity,
                        'lines': (method.start_line, method.end_line)
                    })
            
            if class_related or related_methods:
                results.append({
                    'type': 'class',
                    'name': class_info.name,
                    'docstring': class_info.docstring,
                    'base_classes': class_info.base_classes,
                    'methods': related_methods,
                    'lines': (class_info.start_line, class_info.end_line)
                })
        
        # Check standalone functions
        for func in module_info.functions:
            if is_related(func.docstring):
                results.append({
                    'type': 'function',
                    'name': func.name,
                    'docstring': func.docstring,
                    'params': func.params,
                    'calls': func.calls,
                    'complexity': func.complexity,
                    'lines': (func.start_line, func.end_line)
                })
        
        return results

class BusinessLogicSearcher:
    def __init__(self, search_engine, code_analyzer):
        self.search_engine = search_engine
        self.code_analyzer = code_analyzer
        self.logger = logging.getLogger(__name__)
    
    def search_business_logic(self, feature_description: str, language_filter: str = 'python') -> List[Dict]:
        """Search for business logic related to a specific feature."""
        # First use search engine to find relevant files
        search_results = self.search_engine.search(feature_description, k=5, language_filter=language_filter)
        
        all_results = []
        analyzed_files = set()
        
        for result in search_results:
            file_path = str(Path(result.snippet.repo_path) / result.snippet.file_path)
            
            # Skip if file was already analyzed
            if file_path in analyzed_files:
                continue
                
            analyzed_files.add(file_path)
            
            # Only analyze Python files
            if not file_path.endswith('.py'):
                continue
                
            # Analyze the file
            module_info = self.code_analyzer.analyze_file(file_path)
            
            # Find business logic related to the feature
            logic_results = self.code_analyzer.find_business_logic(module_info, feature_description)
            
            if logic_results:
                all_results.append({
                    'file': file_path,
                    'score': result.score,
                    'elements': logic_results
                })
        
        return all_results
