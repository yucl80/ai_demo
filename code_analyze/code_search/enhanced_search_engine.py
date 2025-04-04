from typing import List, Dict, Any, Optional, Set
import os
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
from .layered_index import LayeredIndex
from .incremental_indexer import IncrementalIndexer
from .model_fusion import ModelFusion
import ast
from collections import defaultdict

@dataclass
class SearchConfig:
    """搜索配置"""
    min_score: float = 0.5
    max_results: int = 10
    language_filter: Optional[str] = None
    search_layers: List[str] = None  # 指定搜索的索引层，如['semantic', 'text', 'dependency']
    include_context: bool = True
    include_symbols: bool = True
    include_dependencies: bool = True
    
class SearchResult:
    def __init__(self, file_path: str, score: float, context: str, metadata: Dict[str, Any]):
        self.file_path = file_path
        self.score = score
        self.context = context
        self.metadata = metadata

class EnhancedSearchEngine:
    def __init__(self, 
                 watch_dirs: List[str],
                 models_config: List[Dict[str, Any]] = None,
                 cache_dir: str = None):
        """
        初始化增强版搜索引擎
        
        Args:
            watch_dirs: 要监控的目录列表
            models_config: 模型配置列表，用于多模型融合
            cache_dir: 缓存目录，用于保存索引状态
        """
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 初始化缓存目录
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # 初始化默认模型配置
        if models_config is None:
            models_config = [
                {
                    'name': 'sentence-bert',
                    'type': 'sentence_transformers',
                    'model_name': 'all-MiniLM-L6-v2',
                    'weight': 0.4
                },
                {
                    'name': 'code-bert',
                    'type': 'huggingface',
                    'model_name': 'microsoft/codebert-base',
                    'weight': 0.4
                },
                {
                    'name': 'tfidf',
                    'type': 'tfidf',
                    'weight': 0.2
                }
            ]
            
        # 初始化组件
        self.model_fusion = ModelFusion(models_config)
        self.layered_index = LayeredIndex()
        self.incremental_indexer = IncrementalIndexer(self.layered_index, watch_dirs)
        
        # 加载缓存的状态
        if cache_dir and os.path.exists(os.path.join(cache_dir, 'index')):
            self.load_state()
        else:
            # 构建初始索引
            self.incremental_indexer.build_initial_index()
            
        # 启动文件监控
        self.incremental_indexer.start_watching()
        
    def search(self, query: str, config: SearchConfig = None) -> List[SearchResult]:
        """
        执行增强搜索，整合多个层次的搜索结果
        """
        if config is None:
            config = SearchConfig()
            
        # 1. 基础搜索
        basic_results = self._basic_search(query, config)
        
        # 2. AST分析增强
        ast_results = self._ast_enhanced_search(query)
        
        # 3. 语义增强
        semantic_results = self._semantic_enhanced_search(query)
        
        # 4. 整合结果
        combined_results = self._combine_search_results(
            basic_results, 
            ast_results,
            semantic_results,
            weights={
                'basic': 0.3,
                'ast': 0.3,
                'semantic': 0.4
            }
        )
        
        return combined_results[:config.max_results]

    def _basic_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        基础搜索
        """
        results = []
        for layer in config.search_layers or ['all']:
            layer_results = self.layered_index.search(
                query=query,
                layer=layer,
                top_k=config.max_results
            )
            
            # 应用过滤条件
            filtered_results = []
            for result in layer_results:
                # 检查分数阈值
                if result['score'] < config.min_score:
                    continue
                    
                # 检查语言过滤
                metadata = result['metadata']
                if config.language_filter and metadata.get('language') != config.language_filter:
                    continue
                    
                # 根据配置添加额外信息
                if not config.include_context:
                    result.pop('context', None)
                if not config.include_symbols:
                    result.pop('symbols', None)
                if not config.include_dependencies:
                    result.pop('dependencies', None)
                    
                filtered_results.append(SearchResult(
                    file_path=result['file_path'],
                    score=result['score'],
                    context=result.get('context', ''),
                    metadata=result['metadata']
                ))
                
            results.extend(filtered_results)
            
        # 对所有结果按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results

    def _ast_enhanced_search(self, query: str) -> List[SearchResult]:
        """
        基于AST的增强搜索
        """
        results = []
        for file_path in self.incremental_indexer.indexed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                visitor = CodeStructureVisitor()
                visitor.visit(tree)
                
                # 分析控制流
                cfg = self._build_control_flow_graph(tree)
                
                # 分析数据流
                dfg = self._build_data_flow_graph(tree)
                
                # 识别设计模式
                patterns = self._detect_design_patterns(tree)
                
                # 计算相关性得分
                score = self._calculate_structure_similarity(
                    query,
                    visitor.structure_info,
                    cfg,
                    dfg,
                    patterns
                )
                
                if score > self.min_score:
                    results.append(SearchResult(
                        file_path=file_path,
                        score=score,
                        context=visitor.get_context(),
                        metadata={
                            'patterns': patterns,
                            'complexity': visitor.complexity
                        }
                    ))
            except Exception as e:
                self.logger.warning(f"AST analysis failed for {file_path}: {str(e)}")
                
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _semantic_enhanced_search(self, query: str) -> List[SearchResult]:
        """
        语义增强搜索
        """
        results = []
        
        # 1. 提取查询意图
        query_intent = self._extract_query_intent(query)
        
        # 2. 分析代码文档
        for file_path in self.incremental_indexer.indexed_files:
            try:
                doc_info = self._extract_documentation(file_path)
                
                # 分析注释
                comments = self._extract_comments(file_path)
                
                # 分析变更历史
                history = self._analyze_change_history(file_path)
                
                # 计算语义相关性
                score = self._calculate_semantic_similarity(
                    query_intent,
                    doc_info,
                    comments,
                    history
                )
                
                if score > self.min_score:
                    results.append(SearchResult(
                        file_path=file_path,
                        score=score,
                        context=doc_info.summary,
                        metadata={
                            'doc_quality': doc_info.quality_score,
                            'last_modified': history.last_change,
                            'change_frequency': history.frequency
                        }
                    ))
            except Exception as e:
                self.logger.warning(f"Semantic analysis failed for {file_path}: {str(e)}")
                
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _combine_search_results(
        self,
        basic_results: List[SearchResult],
        ast_results: List[SearchResult],
        semantic_results: List[SearchResult],
        weights: Dict[str, float]
    ) -> List[SearchResult]:
        """
        组合多个搜索结果，使用加权得分
        """
        combined_scores = defaultdict(float)
        result_metadata = {}
        
        # 1. 合并基础搜索结果
        for result in basic_results:
            combined_scores[result.file_path] += result.score * weights['basic']
            result_metadata[result.file_path] = result.metadata
            
        # 2. 合并AST分析结果
        for result in ast_results:
            combined_scores[result.file_path] += result.score * weights['ast']
            if result.file_path in result_metadata:
                result_metadata[result.file_path].update(result.metadata)
            else:
                result_metadata[result.file_path] = result.metadata
                
        # 3. 合并语义分析结果
        for result in semantic_results:
            combined_scores[result.file_path] += result.score * weights['semantic']
            if result.file_path in result_metadata:
                result_metadata[result.file_path].update(result.metadata)
            else:
                result_metadata[result.file_path] = result.metadata
                
        # 4. 创建最终结果列表
        final_results = []
        for file_path, score in combined_scores.items():
            final_results.append(SearchResult(
                file_path=file_path,
                score=score,
                context=self._get_best_context(file_path, basic_results, ast_results, semantic_results),
                metadata=result_metadata[file_path]
            ))
            
        return sorted(final_results, key=lambda x: x.score, reverse=True)

    def _build_control_flow_graph(self, tree: ast.AST) -> Dict[str, List[str]]:
        """构建控制流图
        
        Args:
            tree: AST语法树
            
        Returns:
            控制流图，key为节点ID，value为后继节点列表
        """
        cfg = {}
        current_node = None
        
        class CFGVisitor(ast.NodeVisitor):
            def __init__(self):
                self.node_counter = 0
                self.current_node = None
                self.cfg = {}
                
            def get_node_id(self):
                self.node_counter += 1
                return f"node_{self.node_counter}"
                
            def visit_FunctionDef(self, node):
                node_id = self.get_node_id()
                self.cfg[node_id] = []
                prev_node = self.current_node
                self.current_node = node_id
                
                # 访问函数体
                for stmt in node.body:
                    self.visit(stmt)
                    
                self.current_node = prev_node
                
            def visit_If(self, node):
                if_id = self.get_node_id()
                self.cfg[if_id] = []
                
                if self.current_node:
                    self.cfg[self.current_node].append(if_id)
                
                # 访问if分支
                prev_node = self.current_node
                self.current_node = if_id
                for stmt in node.body:
                    self.visit(stmt)
                
                # 访问else分支
                if node.orelse:
                    else_id = self.get_node_id()
                    self.cfg[else_id] = []
                    self.cfg[if_id].append(else_id)
                    self.current_node = else_id
                    for stmt in node.orelse:
                        self.visit(stmt)
                        
                self.current_node = prev_node
                
            def visit_While(self, node):
                while_id = self.get_node_id()
                self.cfg[while_id] = []
                
                if self.current_node:
                    self.cfg[self.current_node].append(while_id)
                
                # 访问循环体
                prev_node = self.current_node
                self.current_node = while_id
                for stmt in node.body:
                    self.visit(stmt)
                    
                # 循环回边
                self.cfg[self.current_node].append(while_id)
                self.current_node = prev_node
                
            def generic_visit(self, node):
                if isinstance(node, ast.stmt):
                    node_id = self.get_node_id()
                    self.cfg[node_id] = []
                    if self.current_node:
                        self.cfg[self.current_node].append(node_id)
                    self.current_node = node_id
                super().generic_visit(node)
                
        visitor = CFGVisitor()
        visitor.visit(tree)
        return visitor.cfg

    def _build_data_flow_graph(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """构建数据流图
        
        Args:
            tree: AST语法树
            
        Returns:
            数据流图，key为变量名，value为依赖变量集合
        """
        dfg = {}
        
        class DFGVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dfg = {}
                self.current_scope = set()
                
            def visit_Assign(self, node):
                # 获取被赋值的变量
                targets = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append(target.id)
                        
                # 分析右侧表达式中的变量依赖
                deps = set()
                class VarCollector(ast.NodeVisitor):
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load):
                            deps.add(node.id)
                            
                collector = VarCollector()
                collector.visit(node.value)
                
                # 更新数据流图
                for target in targets:
                    self.dfg[target] = deps
                    
            def visit_FunctionDef(self, node):
                # 分析函数参数
                params = set()
                for arg in node.args.args:
                    params.add(arg.arg)
                    
                # 保存当前作用域
                prev_scope = self.current_scope
                self.current_scope = params
                
                # 分析函数体
                for stmt in node.body:
                    self.visit(stmt)
                    
                # 恢复作用域
                self.current_scope = prev_scope
                
        visitor = DFGVisitor()
        visitor.visit(tree)
        return visitor.dfg

    def _detect_design_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """识别代码中的设计模式
        
        Args:
            tree: AST语法树
            
        Returns:
            识别出的设计模式列表，每个模式包含类型和相关节点
        """
        patterns = []
        
        class PatternDetector(ast.NodeVisitor):
            def __init__(self):
                self.classes = {}
                self.current_class = None
                
            def visit_ClassDef(self, node):
                class_info = {
                    'name': node.name,
                    'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                    'methods': set(),
                    'fields': set()
                }
                
                prev_class = self.current_class
                self.current_class = class_info
                
                # 分析类成员
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        class_info['methods'].add(stmt.name)
                    elif isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                class_info['fields'].add(target.id)
                                
                self.classes[node.name] = class_info
                self.current_class = prev_class
                
            def detect_patterns(self):
                patterns = []
                
                # 检测单例模式
                for name, info in self.classes.items():
                    if '_instance' in info['fields'] and '__new__' in info['methods']:
                        patterns.append({
                            'type': 'Singleton',
                            'class': name
                        })
                        
                # 检测工厂模式
                for name, info in self.classes.items():
                    if 'create' in info['methods'] or 'factory' in info['methods']:
                        patterns.append({
                            'type': 'Factory',
                            'class': name
                        })
                        
                # 检测观察者模式
                for name, info in self.classes.items():
                    if all(m in info['methods'] for m in ['attach', 'detach', 'notify']):
                        patterns.append({
                            'type': 'Observer',
                            'class': name
                        })
                        
                return patterns
                
        detector = PatternDetector()
        detector.visit(tree)
        return detector.detect_patterns()

    def _calculate_structure_similarity(
        self,
        query: str,
        structure_info: Dict[str, Any],
        cfg: Dict[str, List[str]],
        dfg: Dict[str, Set[str]],
        patterns: List[Dict[str, Any]]
    ) -> float:
        """计算代码结构相似度
        
        Args:
            query: 搜索查询
            structure_info: 代码结构信息
            cfg: 控制流图
            dfg: 数据流图
            patterns: 识别出的设计模式
            
        Returns:
            相似度分数 (0-1)
        """
        score = 0.0
        
        # 1. 基于控制流的相似度 (30%)
        if cfg:
            cfg_complexity = len(cfg)  # 节点数量
            cfg_depth = max(self._calculate_graph_depth(cfg, node) for node in cfg)
            cfg_score = 0.3 * (1.0 / (1 + abs(cfg_complexity - 10)))  # 假设理想复杂度为10
            score += cfg_score
            
        # 2. 基于数据流的相似度 (30%)
        if dfg:
            dfg_complexity = sum(len(deps) for deps in dfg.values())  # 总依赖数
            dfg_score = 0.3 * (1.0 / (1 + abs(dfg_complexity - 5)))  # 假设理想依赖数为5
            score += dfg_score
            
        # 3. 基于设计模式的相似度 (40%)
        if patterns:
            # 将查询分词
            query_terms = set(query.lower().split())
            
            # 计算查询与设计模式的匹配度
            pattern_score = 0.0
            for pattern in patterns:
                pattern_terms = set(pattern['type'].lower().split())
                if pattern_terms & query_terms:  # 如果有重叠词
                    pattern_score += 0.4 / len(patterns)
                    
            score += pattern_score
            
        return min(1.0, score)

    def _calculate_graph_depth(self, graph: Dict[str, List[str]], start_node: str, visited: Set[str] = None) -> int:
        """计算图的深度
        
        Args:
            graph: 图的邻接表表示
            start_node: 起始节点
            visited: 已访问节点集合
            
        Returns:
            从起始节点出发的最大深度
        """
        if visited is None:
            visited = set()
            
        if start_node in visited:
            return 0
            
        visited.add(start_node)
        
        if not graph[start_node]:  # 叶子节点
            return 1
            
        max_depth = 0
        for next_node in graph[start_node]:
            depth = self._calculate_graph_depth(graph, next_node, visited)
            max_depth = max(max_depth, depth)
            
        return 1 + max_depth

    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """提取查询意图
        
        Args:
            query: 搜索查询
            
        Returns:
            查询意图信息，包含意图类型和关键词
        """
        # 定义意图关键词映射
        intent_keywords = {
            'implementation': {'implement', 'how', 'example', 'usage'},
            'definition': {'what', 'define', 'declaration'},
            'bug': {'error', 'bug', 'fix', 'issue'},
            'performance': {'slow', 'performance', 'optimize'},
            'security': {'security', 'vulnerability', 'safe'},
            'test': {'test', 'assert', 'verify'}
        }
        
        # 将查询转换为小写并分词
        query_terms = set(query.lower().split())
        
        # 识别意图
        intents = []
        for intent, keywords in intent_keywords.items():
            if query_terms & keywords:
                intents.append(intent)
                
        # 提取关键词（排除意图关键词）
        all_intent_keywords = set().union(*intent_keywords.values())
        keywords = query_terms - all_intent_keywords
        
        return {
            'intents': intents or ['general'],
            'keywords': list(keywords),
            'original_query': query
        }

    def _extract_documentation(self, file_path: str) -> Dict[str, Any]:
        """提取代码文档信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档信息，包含摘要、质量评分等
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            doc_info = {
                'summary': '',
                'quality_score': 0.0,
                'docstrings': [],
                'todos': []
            }
            
            class DocVisitor(ast.NodeVisitor):
                def visit_Module(self, node):
                    if ast.get_docstring(node):
                        doc_info['docstrings'].append({
                            'type': 'module',
                            'content': ast.get_docstring(node)
                        })
                    self.generic_visit(node)
                    
                def visit_ClassDef(self, node):
                    if ast.get_docstring(node):
                        doc_info['docstrings'].append({
                            'type': 'class',
                            'name': node.name,
                            'content': ast.get_docstring(node)
                        })
                    self.generic_visit(node)
                    
                def visit_FunctionDef(self, node):
                    if ast.get_docstring(node):
                        doc_info['docstrings'].append({
                            'type': 'function',
                            'name': node.name,
                            'content': ast.get_docstring(node)
                        })
                    self.generic_visit(node)
                    
            DocVisitor().visit(tree)
            
            # 生成摘要
            if doc_info['docstrings']:
                doc_info['summary'] = doc_info['docstrings'][0]['content'].split('\n')[0]
                
            # 计算文档质量评分
            doc_info['quality_score'] = self._calculate_doc_quality(doc_info['docstrings'])
            
            return doc_info
            
        except Exception as e:
            self.logger.warning(f"Failed to extract documentation from {file_path}: {str(e)}")
            return {
                'summary': '',
                'quality_score': 0.0,
                'docstrings': [],
                'todos': []
            }

    def _calculate_doc_quality(self, docstrings: List[Dict[str, str]]) -> float:
        """计算文档质量评分
        
        Args:
            docstrings: 文档字符串列表
            
        Returns:
            质量评分 (0-1)
        """
        if not docstrings:
            return 0.0
            
        total_score = 0.0
        
        for doc in docstrings:
            content = doc['content']
            
            # 1. 长度评分 (0.3)
            length_score = min(len(content.split()) / 50, 1.0) * 0.3
            
            # 2. 完整性评分 (0.4)
            completeness_score = 0.0
            if 'Args:' in content: completeness_score += 0.1
            if 'Returns:' in content: completeness_score += 0.1
            if 'Raises:' in content: completeness_score += 0.1
            if 'Example:' in content: completeness_score += 0.1
            
            # 3. 可读性评分 (0.3)
            readability_score = 0.3
            if len(content) > 500: readability_score *= 0.8  # 过长的文档可能不易理解
            
            total_score += length_score + completeness_score + readability_score
            
        return min(1.0, total_score / len(docstrings))

    def _extract_comments(self, file_path: str) -> List[Dict[str, Any]]:
        """提取代码注释
        
        Args:
            file_path: 文件路径
            
        Returns:
            注释列表，每个注释包含内容和位置信息
        """
        comments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            in_block_comment = False
            block_comment_content = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # 处理块注释
                if line.startswith('"""') or line.startswith("'''"):
                    if not in_block_comment:
                        in_block_comment = True
                        block_comment_content = [line]
                    else:
                        in_block_comment = False
                        block_comment_content.append(line)
                        comments.append({
                            'type': 'block',
                            'content': '\n'.join(block_comment_content),
                            'line': i - len(block_comment_content) + 1
                        })
                elif in_block_comment:
                    block_comment_content.append(line)
                    
                # 处理行注释
                elif line.startswith('#'):
                    comments.append({
                        'type': 'line',
                        'content': line[1:].strip(),
                        'line': i + 1
                    })
                    
            return comments
            
        except Exception as e:
            self.logger.warning(f"Failed to extract comments from {file_path}: {str(e)}")
            return []

    def _analyze_change_history(self, file_path: str) -> Dict[str, Any]:
        """分析文件变更历史
        
        Args:
            file_path: 文件路径
            
        Returns:
            变更历史信息
        """
        try:
            import os
            import time
            
            stat = os.stat(file_path)
            
            return {
                'last_change': stat.st_mtime,
                'created': stat.st_ctime,
                'size': stat.st_size,
                'frequency': 1.0  # 简化处理，实际应该通过版本控制系统获取
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze change history for {file_path}: {str(e)}")
            return {
                'last_change': 0,
                'created': 0,
                'size': 0,
                'frequency': 0
            }

    def _calculate_semantic_similarity(
        self,
        query_intent: Dict[str, Any],
        doc_info: Dict[str, Any],
        comments: List[Dict[str, Any]],
        history: Dict[str, Any]
    ) -> float:
        """计算语义相似度
        
        Args:
            query_intent: 查询意图
            doc_info: 文档信息
            comments: 注释信息
            history: 变更历史
            
        Returns:
            相似度分数 (0-1)
        """
        score = 0.0
        
        # 1. 基于文档的相似度 (40%)
        if doc_info['docstrings']:
            # 将所有文档内容合并
            doc_text = ' '.join(d['content'] for d in doc_info['docstrings'])
            
            # 计算关键词匹配度
            matches = sum(1 for kw in query_intent['keywords'] if kw.lower() in doc_text.lower())
            doc_score = 0.4 * (matches / len(query_intent['keywords']) if query_intent['keywords'] else 0)
            score += doc_score
            
        # 2. 基于注释的相似度 (30%)
        if comments:
            comment_text = ' '.join(c['content'] for c in comments)
            matches = sum(1 for kw in query_intent['keywords'] if kw.lower() in comment_text.lower())
            comment_score = 0.3 * (matches / len(query_intent['keywords']) if query_intent['keywords'] else 0)
            score += comment_score
            
        # 3. 基于意图的相似度 (20%)
        if 'bug' in query_intent['intents'] and doc_info['todos']:
            score += 0.2  # 如果在查找bug且有TODO标记
        elif 'implementation' in query_intent['intents'] and doc_info['quality_score'] > 0.7:
            score += 0.2  # 如果在查找实现且文档质量高
            
        # 4. 基于时间的相似度 (10%)
        if history['last_change'] > 0:
            # 最近修改的文件得分高
            time_factor = 1.0 / (1 + (time.time() - history['last_change']) / (30 * 24 * 3600))  # 30天衰减
            score += 0.1 * time_factor
            
        return min(1.0, score)

    def _get_best_context(
        self,
        file_path: str,
        basic_results: List[SearchResult],
        ast_results: List[SearchResult],
        semantic_results: List[SearchResult]
    ) -> str:
        """获取最佳上下文
        
        Args:
            file_path: 文件路径
            basic_results: 基础搜索结果
            ast_results: AST分析结果
            semantic_results: 语义分析结果
            
        Returns:
            最佳上下文描述
        """
        contexts = []
        
        # 1. 收集所有结果中的上下文
        for result in basic_results + ast_results + semantic_results:
            if result.file_path == file_path and result.context:
                contexts.append(result.context)
                
        if not contexts:
            return ""
            
        # 2. 选择最长的上下文作为基础
        base_context = max(contexts, key=len)
        
        # 3. 如果有其他上下文包含不同的信息，进行合并
        final_context = base_context
        for ctx in contexts:
            if ctx != base_context and ctx not in base_context:
                final_context += f"\n{ctx}"
                
        return final_context

    def add_feedback(self, feedback_data: List[Dict[str, Any]]):
        """添加搜索反馈，用于优化模型权重"""
        self.model_fusion.update_weights(feedback_data)
        
        if self.cache_dir:
            self.model_fusion.save_weights(
                os.path.join(self.cache_dir, 'model_weights.json')
            )
            
    def save_state(self):
        """保存搜索引擎状态"""
        if not self.cache_dir:
            return
            
        # 保存索引状态
        self.incremental_indexer.save_state(
            os.path.join(self.cache_dir, 'index')
        )
        
        # 保存模型权重
        self.model_fusion.save_weights(
            os.path.join(self.cache_dir, 'model_weights.json')
        )
        
    def load_state(self):
        """加载搜索引擎状态"""
        if not self.cache_dir:
            return
            
        # 加载索引状态
        self.incremental_indexer.load_state(
            os.path.join(self.cache_dir, 'index')
        )
        
        # 加载模型权重
        weights_path = os.path.join(self.cache_dir, 'model_weights.json')
        if os.path.exists(weights_path):
            self.model_fusion.load_weights(weights_path)
            
    def __del__(self):
        """清理资源"""
        # 停止文件监控
        self.incremental_indexer.stop_watching()
        
        # 保存状态
        if self.cache_dir:
            self.save_state()

class CodeStructureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structure_info = {}
        self.complexity = 0

    def visit(self, node):
        # TODO: 实现AST节点访问
        pass

    def get_context(self):
        # TODO: 实现上下文获取
        pass
