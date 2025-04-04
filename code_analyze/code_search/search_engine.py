import os
from typing import List, Dict, Optional, Set, Tuple
import faiss
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
from functools import lru_cache
import re

@dataclass
class CodeSnippet:
    content: str
    file_path: str
    repo_path: str
    start_line: int
    end_line: int
    language: str
    context: Optional[str] = None
    symbols: Optional[List[str]] = None

@dataclass
class SearchResult:
    snippet: CodeSnippet
    score: float
    
class EnhancedIndex:
    """Enhanced FAISS index with optimizations for code search."""
    def __init__(self, dimension: int, nlist: int = 100):
        # Use IVF (Inverted File Index) for faster search
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        self.is_trained = False
        
    def train(self, vectors: np.ndarray):
        if not self.is_trained and len(vectors) > 0:
            self.index.train(vectors)
            self.is_trained = True
    
    def add(self, vectors: np.ndarray):
        if not self.is_trained:
            self.train(vectors)
        self.index.add(vectors)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Adjust nprobe based on k to balance speed and recall
        self.index.nprobe = min(20, k * 2)
        return self.index.search(query, k)
    
    def save(self, path: str):
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        self.index = faiss.read_index(path)
        self.is_trained = True

class CodeSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.index = None
        self.snippets: List[CodeSnippet] = []
        self.index_to_snippet: Dict[int, CodeSnippet] = {}
        self.indexed_files: Set[str] = set()
        self.symbol_cache: Dict[str, Set[str]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize text splitters for different languages
        self.language_splitters = {
            'python': RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'javascript': RecursiveCharacterTextSplitter.from_language(
                language=Language.JS,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'java': RecursiveCharacterTextSplitter.from_language(
                language=Language.JAVA,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'cpp': RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'rust': RecursiveCharacterTextSplitter.from_language(
                language=Language.RUST,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'go': RecursiveCharacterTextSplitter.from_language(
                language=Language.GO,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'default': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
        }
        
        # Initialize symbol extractors
        self.symbol_patterns = {
            'python': [
                r'class\s+(\w+)',
                r'def\s+(\w+)',
                r'(\w+)\s*=\s*',
            ],
            'javascript': [
                r'class\s+(\w+)',
                r'function\s+(\w+)',
                r'const\s+(\w+)\s*=',
                r'let\s+(\w+)\s*=',
            ],
            'java': [
                r'class\s+(\w+)',
                r'(public|private|protected)\s+\w+\s+(\w+)\s*\(',
                r'interface\s+(\w+)',
            ],
        }
        
        # Setup caching
        self._get_code_embedding = lru_cache(maxsize=cache_size)(self._compute_embedding)
        
    def _compute_embedding(self, code: str) -> np.ndarray:
        """Compute embedding for a code snippet."""
        try:
            embeddings = self.embeddings.embed_documents([code])
            return np.array(embeddings[0])
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(384)

    def _extract_symbols(self, content: str, language: str) -> Set[str]:
        """Extract code symbols (class names, function names, etc.) from content."""
        patterns = self.symbol_patterns.get(language, [])
        symbols = set()
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Get the last group (actual symbol name)
                symbol = match.groups()[-1]
                if symbol:
                    symbols.add(symbol)
                    
        return symbols

    def _get_file_context(self, file_path: str, start_line: int, end_line: int, context_lines: int = 3) -> str:
        """Get surrounding context for a code snippet."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            start_ctx = max(0, start_line - context_lines)
            end_ctx = min(len(lines), end_line + context_lines)
            
            return ''.join(lines[start_ctx:end_ctx])
        except Exception as e:
            self.logger.error(f"Error getting context: {str(e)}")
            return ""

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.ts': 'javascript',
            '.rb': 'ruby',
            '.php': 'php',
        }
        return language_map.get(ext, 'default')

    def _split_code_to_snippets(self, content: str, language: str) -> List[Document]:
        """Split code content into snippets using LangChain's text splitter."""
        splitter = self.language_splitters.get(language, self.language_splitters['default'])
        return splitter.create_documents([content])

    def add_repository(self, repo_path: str, file_patterns: Optional[List[str]] = None):
        """Index all code files in a repository."""
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.java', '*.cpp', '*.c', '*.rs', '*.go', '*.ts']
            
        repo_path = Path(repo_path)
        new_snippets = []
        
        def process_file(file_path: Path):
            if str(file_path) in self.indexed_files:
                return []
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                language = self._detect_language(str(file_path))
                documents = self._split_code_to_snippets(content, language)
                
                # Extract symbols from the entire file
                file_symbols = self._extract_symbols(content, language)
                
                file_snippets = []
                for doc in documents:
                    start_line = doc.metadata.get('start_line', 0)
                    end_line = doc.metadata.get('end_line', 0)
                    
                    # Get context for the snippet
                    context = self._get_file_context(
                        str(file_path), 
                        start_line, 
                        end_line
                    )
                    
                    snippet = CodeSnippet(
                        content=doc.page_content,
                        file_path=str(file_path.relative_to(repo_path)),
                        repo_path=str(repo_path),
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        context=context,
                        symbols=list(file_symbols)
                    )
                    file_snippets.append(snippet)
                    
                return file_snippets
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                return []

        # Collect all files matching patterns
        all_files = []
        for pattern in file_patterns:
            all_files.extend(repo_path.rglob(pattern))

        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            for file_snippets in executor.map(process_file, all_files):
                new_snippets.extend(file_snippets)
                
        if not new_snippets:
            return

        # Generate embeddings for new snippets
        embeddings = []
        for snippet in new_snippets:
            # Create enhanced embedding by combining code and symbols
            enhanced_content = f"{snippet.content}\n{' '.join(snippet.symbols or [])}"
            embedding = self._get_code_embedding(enhanced_content)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Update or create FAISS index
        if self.index is None:
            self.index = EnhancedIndex(embeddings.shape[1])
            
        self.index.add(embeddings)
        
        # Update snippets and mapping
        start_idx = len(self.snippets)
        for i, snippet in enumerate(new_snippets):
            self.index_to_snippet[start_idx + i] = snippet
            self.indexed_files.add(str(Path(snippet.repo_path) / snippet.file_path))
            
        self.snippets.extend(new_snippets)
        
    def search(self, 
              query: str, 
              k: int = 5, 
              language_filter: Optional[str] = None,
              min_score: float = 0.5
              ) -> List[SearchResult]:
        """Search for code snippets similar to the query with enhanced filtering."""
        if self.index is None or len(self.snippets) == 0:
            return []
        
        # Enhance query with potential code symbols
        symbols = set(re.findall(r'\b\w+\b', query))
        enhanced_query = query
        for symbol in symbols:
            if any(symbol in s.symbols for s in self.snippets if s.symbols):
                enhanced_query = f"{enhanced_query} {symbol}"
            
        query_embedding = self._get_code_embedding(enhanced_query)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k * 2)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            score = 1.0 / (1.0 + distance)
            if score < min_score:
                continue
                
            snippet = self.index_to_snippet[idx]
            
            # Apply language filter if specified
            if language_filter and snippet.language != language_filter:
                continue
                
            results.append(SearchResult(
                snippet=snippet,
                score=score
            ))
            
            if len(results) >= k:
                break
                
        return results

    def save_index(self, path: str):
        """Save the search index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
            
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        self.index.save(os.path.join(path, "code_search.index"))
        
        # Save snippets and mapping
        metadata = {
            "snippets": [
                {
                    "content": s.content,
                    "file_path": s.file_path,
                    "repo_path": s.repo_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "language": s.language,
                    "context": s.context,
                    "symbols": s.symbols
                }
                for s in self.snippets
            ],
            "indexed_files": list(self.indexed_files)
        }
        
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
    def load_index(self, path: str):
        """Load the search index and metadata from disk."""
        if not os.path.exists(path):
            raise ValueError(f"Index path {path} does not exist")
            
        # Load FAISS index
        self.index = EnhancedIndex(384)  # Default dimension for all-MiniLM-L6-v2
        self.index.load(os.path.join(path, "code_search.index"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
            
        self.snippets = []
        self.index_to_snippet = {}
        
        for i, s in enumerate(metadata["snippets"]):
            snippet = CodeSnippet(
                content=s["content"],
                file_path=s["file_path"],
                repo_path=s["repo_path"],
                start_line=s["start_line"],
                end_line=s["end_line"],
                language=s["language"],
                context=s.get("context"),
                symbols=s.get("symbols", [])
            )
            self.snippets.append(snippet)
            self.index_to_snippet[i] = snippet
            
        self.indexed_files = set(metadata["indexed_files"])
