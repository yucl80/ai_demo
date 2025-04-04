import os
import hashlib
from typing import Dict, List, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .layered_index import LayeredIndex

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer):
        self.indexer = indexer
        
    def on_modified(self, event):
        if not event.is_directory:
            self.indexer.handle_file_change(event.src_path, 'modified')
            
    def on_created(self, event):
        if not event.is_directory:
            self.indexer.handle_file_change(event.src_path, 'created')
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.indexer.handle_file_change(event.src_path, 'deleted')

class IncrementalIndexer:
    def __init__(self, index: LayeredIndex, watch_dirs: List[str]):
        self.index = index
        self.watch_dirs = watch_dirs
        self.file_hashes = {}  # 存储文件哈希值
        self.observer = Observer()
        self.event_handler = FileChangeHandler(self)
        
    def start_watching(self):
        """开始监控文件变化"""
        for dir_path in self.watch_dirs:
            self.observer.schedule(self.event_handler, dir_path, recursive=True)
        self.observer.start()
        
    def stop_watching(self):
        """停止监控文件变化"""
        self.observer.stop()
        self.observer.join()
        
    def compute_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def handle_file_change(self, file_path: str, event_type: str):
        """处理文件变更"""
        if event_type == 'deleted':
            if file_path in self.file_hashes:
                self.index.remove_document(file_path)
                del self.file_hashes[file_path]
        else:
            # 对于新建或修改的文件
            new_hash = self.compute_file_hash(file_path)
            old_hash = self.file_hashes.get(file_path)
            
            if new_hash != old_hash:
                # 文件内容发生变化，更新索引
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 获取文件元数据
                metadata = {
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'last_modified': os.path.getmtime(file_path)
                }
                
                # 分析文件依赖关系
                dependencies = self.analyze_dependencies(file_path)
                
                # 分析业务逻辑
                business_logic = self.analyze_business_logic(file_path, content)
                
                # 更新索引
                self.index.update_document(
                    doc_id=file_path,
                    content=content,
                    metadata=metadata,
                    dependencies=dependencies,
                    business_logic=business_logic
                )
                
                # 更新文件哈希值
                self.file_hashes[file_path] = new_hash
                
    def analyze_dependencies(self, file_path: str) -> List[str]:
        """分析文件依赖关系"""
        dependencies = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 分析Python import语句
        if file_path.endswith('.py'):
            import ast
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            dependencies.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            dependencies.append(node.module)
            except:
                pass
                
        return dependencies
        
    def analyze_business_logic(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析文件业务逻辑"""
        business_logic = {
            'functions': [],
            'classes': [],
            'variables': []
        }
        
        if file_path.endswith('.py'):
            import ast
            try:
                tree = ast.parse(content)
                
                # 提取函数信息
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        business_logic['functions'].append({
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'docstring': ast.get_docstring(node)
                        })
                    elif isinstance(node, ast.ClassDef):
                        business_logic['classes'].append({
                            'name': node.name,
                            'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                            'docstring': ast.get_docstring(node)
                        })
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                business_logic['variables'].append(target.id)
            except:
                pass
                
        return business_logic
        
    def build_initial_index(self):
        """构建初始索引"""
        for dir_path in self.watch_dirs:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.handle_file_change(file_path, 'created')
                    
    def save_state(self, save_dir: str):
        """保存索引状态"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文件哈希值
        import json
        with open(os.path.join(save_dir, 'file_hashes.json'), 'w') as f:
            json.dump(self.file_hashes, f)
            
        # 保存索引
        self.index.save_index(os.path.join(save_dir, 'index'))
        
    def load_state(self, save_dir: str):
        """加载索引状态"""
        # 加载文件哈希值
        import json
        try:
            with open(os.path.join(save_dir, 'file_hashes.json'), 'r') as f:
                self.file_hashes = json.load(f)
        except FileNotFoundError:
            self.file_hashes = {}
            
        # 加载索引
        self.index.load_index(os.path.join(save_dir, 'index'))
