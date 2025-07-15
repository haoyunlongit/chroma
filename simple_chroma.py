"""
精简版Chroma向量数据库实现
提供基本的向量存储、检索和相似性搜索功能
"""

import numpy as np
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle


class SimpleChroma:
    """精简版Chroma向量数据库"""
    
    def __init__(self, name: str = "default", persist_directory: Optional[str] = None):
        """
        初始化向量数据库
        
        Args:
            name: 数据库名称
            persist_directory: 持久化目录路径
        """
        self.name = name
        self.persist_directory = persist_directory
        
        # 存储向量数据
        self._embeddings: Dict[str, np.ndarray] = {}
        self._documents: Dict[str, str] = {}
        self._metadatas: Dict[str, Dict] = {}
        self._ids: List[str] = []
        
        # TF-IDF向量化器用于文本向量化
        self._vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._vectorizer_fitted = False
        
        # 如果指定了持久化目录，尝试加载已有数据
        if persist_directory and os.path.exists(self._get_persist_path()):
            self.load()
    
    def _get_persist_path(self) -> str:
        """获取持久化文件路径"""
        if not self.persist_directory:
            raise ValueError("未设置持久化目录")
        return os.path.join(self.persist_directory, f"{self.name}_db.pkl")
    
    def add(
        self, 
        documents: List[str], 
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        添加文档到数据库
        
        Args:
            documents: 文档文本列表
            embeddings: 预计算的向量列表（可选）
            metadatas: 元数据列表（可选）
            ids: 文档ID列表（可选）
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        # 生成ID（如果未提供）
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if len(ids) != len(documents):
            raise ValueError("ID数量必须与文档数量相等")
        
        # 生成或使用提供的向量
        if embeddings is None:
            embeddings = self._create_embeddings(documents)
        else:
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings]
        
        # 处理元数据
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 存储数据
        for i, doc_id in enumerate(ids):
            if doc_id in self._ids:
                # 更新已存在的文档
                self._embeddings[doc_id] = embeddings[i]
                self._documents[doc_id] = documents[i]
                self._metadatas[doc_id] = metadatas[i]
            else:
                # 添加新文档
                self._ids.append(doc_id)
                self._embeddings[doc_id] = embeddings[i]
                self._documents[doc_id] = documents[i]
                self._metadatas[doc_id] = metadatas[i]
        
        print(f"成功添加 {len(documents)} 个文档到数据库")
    
    def _create_embeddings(self, documents: List[str]) -> List[np.ndarray]:
        """
        使用TF-IDF创建文档向量
        
        Args:
            documents: 文档列表
            
        Returns:
            向量列表
        """
        if not self._vectorizer_fitted:
            # 训练向量化器
            all_docs = list(self._documents.values()) + documents
            self._vectorizer.fit(all_docs)
            self._vectorizer_fitted = True
            
            # 重新计算已存在文档的向量
            if self._documents:
                for doc_id, doc in self._documents.items():
                    self._embeddings[doc_id] = self._vectorizer.transform([doc]).toarray()[0]
        
        # 转换新文档
        embeddings = self._vectorizer.transform(documents).toarray()
        return [emb.astype(np.float32) for emb in embeddings]
    
    def query(
        self, 
        query_texts: List[str], 
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        查询相似文档
        
        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            
        Returns:
            查询结果字典
        """
        if not self._ids:
            return {
                'ids': [[] for _ in query_texts],
                'distances': [[] for _ in query_texts],
                'documents': [[] for _ in query_texts],
                'metadatas': [[] for _ in query_texts]
            }
        
        results = {
            'ids': [],
            'distances': [],
            'documents': [],
            'metadatas': []
        }
        
        for query_text in query_texts:
            # 创建查询向量
            if self._vectorizer_fitted:
                query_embedding = self._vectorizer.transform([query_text]).toarray()[0]
            else:
                # 如果还没有训练向量化器，返回空结果
                results['ids'].append([])
                results['distances'].append([])
                results['documents'].append([])
                results['metadatas'].append([])
                continue
            
            # 计算相似度
            similarities = []
            valid_ids = []
            
            for doc_id in self._ids:
                # 应用元数据过滤
                if where and not self._match_metadata(self._metadatas[doc_id], where):
                    continue
                
                similarity = cosine_similarity(
                    [query_embedding], 
                    [self._embeddings[doc_id]]
                )[0][0]
                similarities.append(similarity)
                valid_ids.append(doc_id)
            
            # 排序并获取top-k结果
            if similarities:
                sorted_indices = np.argsort(similarities)[::-1][:n_results]
                
                query_ids = [valid_ids[i] for i in sorted_indices]
                query_distances = [1 - similarities[i] for i in sorted_indices]  # 转换为距离
                query_documents = [self._documents[doc_id] for doc_id in query_ids]
                query_metadatas = [self._metadatas[doc_id] for doc_id in query_ids]
            else:
                query_ids = []
                query_distances = []
                query_documents = []
                query_metadatas = []
            
            results['ids'].append(query_ids)
            results['distances'].append(query_distances)
            results['documents'].append(query_documents)
            results['metadatas'].append(query_metadatas)
        
        return results
    
    def _match_metadata(self, metadata: Dict, where: Dict) -> bool:
        """
        检查元数据是否匹配过滤条件
        
        Args:
            metadata: 文档元数据
            where: 过滤条件
            
        Returns:
            是否匹配
        """
        for key, value in where.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]) -> None:
        """
        删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
        """
        deleted_count = 0
        for doc_id in ids:
            if doc_id in self._ids:
                self._ids.remove(doc_id)
                del self._embeddings[doc_id]
                del self._documents[doc_id]
                del self._metadatas[doc_id]
                deleted_count += 1
        
        print(f"成功删除 {deleted_count} 个文档")
    
    def count(self) -> int:
        """返回数据库中文档数量"""
        return len(self._ids)
    
    def peek(self, limit: int = 10) -> Dict[str, List]:
        """
        查看数据库中的前几个文档
        
        Args:
            limit: 查看数量限制
            
        Returns:
            文档信息字典
        """
        preview_ids = self._ids[:limit]
        return {
            'ids': preview_ids,
            'documents': [self._documents[doc_id] for doc_id in preview_ids],
            'metadatas': [self._metadatas[doc_id] for doc_id in preview_ids]
        }
    
    def persist(self) -> None:
        """持久化数据库到磁盘"""
        if not self.persist_directory:
            raise ValueError("未设置持久化目录")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        data = {
            'name': self.name,
            'embeddings': self._embeddings,
            'documents': self._documents,
            'metadatas': self._metadatas,
            'ids': self._ids,
            'vectorizer': self._vectorizer,
            'vectorizer_fitted': self._vectorizer_fitted
        }
        
        with open(self._get_persist_path(), 'wb') as f:
            pickle.dump(data, f)
        
        print(f"数据库已保存到 {self._get_persist_path()}")
    
    def load(self) -> None:
        """从磁盘加载数据库"""
        if not self.persist_directory:
            raise ValueError("未设置持久化目录")
        
        persist_path = self._get_persist_path()
        if not os.path.exists(persist_path):
            print(f"持久化文件不存在: {persist_path}")
            return
        
        with open(persist_path, 'rb') as f:
            data = pickle.load(f)
        
        self.name = data['name']
        self._embeddings = data['embeddings']
        self._documents = data['documents']
        self._metadatas = data['metadatas']
        self._ids = data['ids']
        self._vectorizer = data['vectorizer']
        self._vectorizer_fitted = data['vectorizer_fitted']
        
        print(f"从 {persist_path} 加载了 {len(self._ids)} 个文档")


class ChromaClient:
    """Chroma客户端，用于管理多个数据库"""
    
    def __init__(self, persist_directory: Optional[str] = "./chroma_db"):
        """
        初始化客户端
        
        Args:
            persist_directory: 持久化目录
        """
        self.persist_directory = persist_directory
        self._collections: Dict[str, SimpleChroma] = {}
        
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
    
    def create_collection(
        self, 
        name: str, 
        get_or_create: bool = False
    ) -> SimpleChroma:
        """
        创建新的集合（数据库）
        
        Args:
            name: 集合名称
            get_or_create: 如果集合已存在，是否返回现有集合
            
        Returns:
            向量数据库实例
        """
        if name in self._collections:
            if get_or_create:
                return self._collections[name]
            else:
                raise ValueError(f"集合 '{name}' 已存在")
        
        collection = SimpleChroma(name=name, persist_directory=self.persist_directory)
        self._collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> SimpleChroma:
        """
        获取现有集合
        
        Args:
            name: 集合名称
            
        Returns:
            向量数据库实例
        """
        if name not in self._collections:
            # 尝试从磁盘加载
            collection = SimpleChroma(name=name, persist_directory=self.persist_directory)
            if collection.count() > 0:  # 如果成功加载了数据
                self._collections[name] = collection
            else:
                raise ValueError(f"集合 '{name}' 不存在")
        
        return self._collections[name]
    
    def delete_collection(self, name: str) -> None:
        """
        删除集合
        
        Args:
            name: 集合名称
        """
        if name in self._collections:
            del self._collections[name]
        
        # 删除持久化文件
        if self.persist_directory:
            persist_path = os.path.join(self.persist_directory, f"{name}_db.pkl")
            if os.path.exists(persist_path):
                os.remove(persist_path)
                print(f"已删除集合文件: {persist_path}")
    
    def list_collections(self) -> List[str]:
        """列出所有集合名称"""
        collection_names = list(self._collections.keys())
        
        # 同时检查持久化目录中的文件
        if self.persist_directory and os.path.exists(self.persist_directory):
            for filename in os.listdir(self.persist_directory):
                if filename.endswith('_db.pkl'):
                    name = filename[:-7]  # 移除 '_db.pkl' 后缀
                    if name not in collection_names:
                        collection_names.append(name)
        
        return collection_names


# 便捷函数
def create_client(persist_directory: str = "./chroma_db") -> ChromaClient:
    """创建Chroma客户端"""
    return ChromaClient(persist_directory=persist_directory) 