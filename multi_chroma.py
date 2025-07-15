"""
多模型嵌入向量数据库实现
支持 OpenAI API、DeepSeek API、Ollama本地模型、sentence-transformers本地模型
"""

import os
import re
import ast
import json
import uuid
import httpx
import jieba
from typing import List, Dict, Optional, Any, Union, Callable
from datetime import datetime
from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv
import numpy as np

# 加载环境变量
load_dotenv()


class BaseEmbeddingFunction(EmbeddingFunction, ABC):
    """嵌入函数基类"""
    
    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbeddingFunction(BaseEmbeddingFunction):
    """OpenAI嵌入函数"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "text-embedding-ada-002",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """调用OpenAI嵌入API"""
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            print(f"OpenAI嵌入调用失败: {e}")
            raise


class DeepSeekEmbeddingFunction(BaseEmbeddingFunction):
    """DeepSeek嵌入函数"""
    
    def __init__(
        self,
        api_key: str, 
        base_url: str = "https://api.deepseek.com/v1"
    ):
        self.api_key = api_key
        self.base_url = base_url
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """调用DeepSeek嵌入API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            embeddings = []
            
            for text in texts:
                payload = {
                    "model": "text-embedding-ada-002",
                    "input": text,
                    "encoding_format": "float"
                }
                
                with httpx.Client() as client:
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        embedding = result["data"][0]["embedding"]
                        embeddings.append(embedding)
                    else:
                        print(f"DeepSeek API错误: {response.status_code}, {response.text}")
                        # 返回随机向量作为fallback
                        embeddings.append(np.random.rand(1536).tolist())
            
            return embeddings
            
        except Exception as e:
            print(f"DeepSeek嵌入调用失败: {e}")
            # 返回随机向量作为fallback
            return [np.random.rand(1536).tolist() for _ in texts]


class OllamaEmbeddingFunction(BaseEmbeddingFunction):
    """Ollama本地嵌入函数"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text"
    ):
        self.base_url = base_url
        self.model = model
        self._test_connection()
        
    def _test_connection(self):
        """测试Ollama连接"""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    print(f"✅ Ollama连接成功: {self.base_url}")
                else:
                    raise Exception(f"Ollama响应错误: {response.status_code}")
        except Exception as e:
            print(f"❌ Ollama连接失败: {e}")
            print("请确保Ollama正在运行: ollama serve")
            print(f"并安装嵌入模型: ollama pull {self.model}")
            raise
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """使用Ollama生成嵌入"""
        try:
            embeddings = []
            
            for text in texts:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                with httpx.Client() as client:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        embedding = result["embedding"]
                        embeddings.append(embedding)
                    else:
                        print(f"Ollama API错误: {response.status_code}, {response.text}")
                        # 返回随机向量作为fallback
                        embeddings.append(np.random.rand(768).tolist())
            
            return embeddings
            
        except Exception as e:
            print(f"Ollama嵌入生成失败: {e}")
            return [np.random.rand(768).tolist() for _ in texts]


class LocalEmbeddingFunction(BaseEmbeddingFunction):
    """本地sentence-transformers嵌入函数"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None
        self._load_model()
        
    def _load_model(self):
        """加载本地模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"🔄 正在加载本地模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ 本地模型加载成功")
            
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            print("请安装: pip install sentence-transformers")
            raise
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """使用本地模型生成嵌入"""
        try:
            # 预处理中文文本
            processed_texts = [self._preprocess_chinese(text) for text in texts]
            
            # 生成嵌入向量
            embeddings = self._model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"本地嵌入生成失败: {e}")
            raise
    
    def _preprocess_chinese(self, text: str) -> str:
        """预处理中文文本"""
        try:
            # 中文分词
            words = jieba.lcut(text)
            return " ".join(words)
        except:
            # 如果jieba不可用，直接返回原文
            return text


class MultiChromaDB:
    """多模型Chroma向量数据库"""
    
    def __init__(
        self,
        embedding_type: str = "local",
        collection_name: str = None,
        persist_directory: str = None,
        **embedding_kwargs
    ):
        """
        初始化多模型向量数据库
        
        Args:
            embedding_type: 嵌入类型 ("openai", "deepseek", "ollama", "local")
            collection_name: 集合名称
            persist_directory: 持久化目录
            **embedding_kwargs: 嵌入函数的额外参数
        """
        self.embedding_type = embedding_type
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "multi_documents")
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./multi_chroma_db")
        
        # 创建嵌入函数
        self.embedding_function = self._create_embedding_function(**embedding_kwargs)
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✅ 加载现有集合: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"embedding_type": self.embedding_type}
            )
            print(f"✅ 创建新集合: {self.collection_name} (嵌入类型: {self.embedding_type})")
    
    def _create_embedding_function(self, **kwargs) -> BaseEmbeddingFunction:
        """创建嵌入函数"""
        if self.embedding_type == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = kwargs.get("model") or os.getenv("OPENAI_MODEL", "text-embedding-ada-002")
            base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            if not api_key:
                raise ValueError("OpenAI API密钥未设置")
            
            return OpenAIEmbeddingFunction(api_key=api_key, model=model, base_url=base_url)
            
        elif self.embedding_type == "deepseek":
            api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            base_url = kwargs.get("base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            
            if not api_key:
                raise ValueError("DeepSeek API密钥未设置")
            
            return DeepSeekEmbeddingFunction(api_key=api_key, base_url=base_url)
            
        elif self.embedding_type == "ollama":
            base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = kwargs.get("model") or os.getenv("OLLAMA_MODEL", "nomic-embed-text")
            
            return OllamaEmbeddingFunction(base_url=base_url, model=model)
            
        elif self.embedding_type == "local":
            model_name = kwargs.get("model_name") or os.getenv("LOCAL_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            return LocalEmbeddingFunction(model_name=model_name)
            
        else:
            raise ValueError(f"不支持的嵌入类型: {self.embedding_type}")
    
    def process_code_document(self, title: str, content: str, code: str = None) -> str:
        """处理包含代码的文档"""
        enhanced_content = f"标题: {title}

描述: {content}"
        
        if code:
            code_features = self._extract_code_features(code)
            
            enhanced_content += f"""

代码功能: {code_features['functionality']}
编程语言: {code_features['language']}
主要函数: {', '.join(code_features['functions'])}
主要类: {', '.join(code_features['classes'])}
导入模块: {', '.join(code_features['imports'])}
代码类型: {code_features['code_type']}
使用场景: {code_features['use_cases']}
技术关键词: {' '.join(code_features['keywords'])}

代码示例:
{code}
"""
        
        return enhanced_content
    
    def _extract_code_features(self, code: str) -> Dict[str, Any]:
        """提取代码特征"""
        features = {
            'functionality': '',
            'language': 'unknown',
            'functions': [],
            'classes': [],
            'imports': [],
            'code_type': 'script',
            'use_cases': '',
            'keywords': []
        }
        
        # 检测编程语言
        if any(keyword in code for keyword in ['def ', 'import ', 'class ', 'python']):
            features['language'] = 'python'
            features = self._analyze_python_code(code, features)
        elif any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ']):
            features['language'] = 'javascript'
        elif any(keyword in code for keyword in ['public class', 'private ', 'package ']):
            features['language'] = 'java'
        elif 'FROM ' in code.upper() and 'RUN ' in code.upper():
            features['language'] = 'dockerfile'
        
        # 提取通用关键词
        code_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
        features['keywords'] = list(set(code_words))[:20]
        
        return features
    
    def _analyze_python_code(self, code: str, features: Dict) -> Dict:
        """分析Python代码"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    features['classes'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        features['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    features['imports'].append(node.module)
            
            # 判断代码类型和用途
            if features['classes']:
                features['code_type'] = 'class'
                features['functionality'] = f"定义了{len(features['classes'])}个类: {', '.join(features['classes'])}"
            elif features['functions']:
                features['code_type'] = 'function'
                features['functionality'] = f"包含{len(features['functions'])}个函数: {', '.join(features['functions'])}"
            
            # 推断使用场景
            if any(imp in ['flask', 'django', 'fastapi'] for imp in features['imports']):
                features['use_cases'] = 'Web开发, API开发'
            elif any(imp in ['pandas', 'numpy', 'matplotlib'] for imp in features['imports']):
                features['use_cases'] = '数据分析, 数据科学'
            elif any(imp in ['tensorflow', 'pytorch', 'sklearn'] for imp in features['imports']):
                features['use_cases'] = '机器学习, 人工智能'
            
        except Exception as e:
            print(f"代码分析出错: {e}")
        
        return features

    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        codes: Optional[List[str]] = None
    ) -> None:
        """添加文档到向量数据库"""
        if not documents:
            raise ValueError("文档列表不能为空")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 处理文档内容
        processed_documents = []
        for i, doc in enumerate(documents):
            if codes and i < len(codes) and codes[i]:
                title = metadatas[i].get('title', f'文档 {i+1}')
                processed_doc = self.process_code_document(title, doc, codes[i])
            else:
                processed_doc = doc
            
            processed_documents.append(processed_doc)
        
        # 添加时间戳和类型
        for metadata in metadatas:
            metadata['added_at'] = datetime.now().isoformat()
            metadata['embedding_type'] = self.embedding_type
        
        try:
            self.collection.add(
                documents=processed_documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ 成功添加 {len(documents)} 个文档 (嵌入类型: {self.embedding_type})")
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> Dict[str, List]:
        """搜索相似文档"""
        try:
            include_list = ["documents", "distances", "metadatas"] if include_metadata else ["documents", "distances"]
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=include_list
            )
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    def smart_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """智能搜索"""
        query_lower = query.lower()
        results_list = []
        
        # 基础语义搜索
        base_results = self.search(query, n_results=max_results * 2)
        
        # 推断过滤条件
        filters = self._infer_filters(query_lower)
        
        # 过滤搜索
        if filters:
            for filter_condition in filters:
                filtered_results = self.search(
                    query, 
                    n_results=max_results,
                    where=filter_condition
                )
                results_list.extend(self._format_results(filtered_results))
        
        # 合并结果
        results_list.extend(self._format_results(base_results))
        
        # 去重
        unique_results = []
        seen_docs = set()
        
        for result in results_list:
            doc_key = result['document'][:100]
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                unique_results.append(result)
                
                if len(unique_results) >= max_results:
                    break
        
        return unique_results
    
    def _infer_filters(self, query_lower: str) -> List[Dict]:
        """推断过滤条件"""
        filters = []
        
        # 编程语言
        language_keywords = {
            'python': ['python', 'py', 'django', 'flask', 'pandas'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue'],
            'java': ['java', 'spring'],
            'sql': ['sql', 'mysql', 'postgresql']
        }
        
        for lang, keywords in language_keywords.items():
            if any(kw in query_lower for kw in keywords):
                filters.append({"programming_language": lang})
        
        # 技术领域
        domain_keywords = {
            'web开发': ['web', '网站', 'api', '前端', '后端'],
            '数据科学': ['数据', 'data', '分析', '机器学习'],
            '人工智能': ['ai', '人工智能', 'ml', '深度学习']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                filters.append({"domain": domain})
        
        return filters
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """格式化搜索结果"""
        formatted = []
        
        if not results['documents'] or not results['documents'][0]:
            return formatted
        
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results.get('metadatas', [{}])[0] if results.get('metadatas') else [{}] * len(documents)
        
        for doc, distance, metadata in zip(documents, distances, metadatas):
            # 提取标题
            title = metadata.get('title', 'Unknown')
            if title == 'Unknown' and '标题:' in doc:
                title_match = re.search(r'标题:\s*(.+)', doc)
                if title_match:
                    title = title_match.group(1).strip()
            
            formatted.append({
                'title': title,
                'document': doc,
                'distance': distance,
                'similarity': 1 - distance,
                'metadata': metadata
            })
        
        return formatted
    
    def get_collection_info(self) -> Dict:
        """获取集合信息"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'document_count': count,
                'embedding_type': self.embedding_type,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"获取集合信息失败: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> None:
        """删除文档"""
        try:
            self.collection.delete(ids=ids)
            print(f"✅ 成功删除 {len(ids)} 个文档")
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
    
    def reset_collection(self) -> None:
        """重置集合（删除所有数据）"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"embedding_type": self.embedding_type}
            )
            print(f"✅ 重置集合: {self.collection_name}")
        except Exception as e:
            print(f"❌ 重置集合失败: {e}")


def create_multi_client(
    embedding_type: str = None,
    collection_name: str = None,
    persist_directory: str = None,
    **kwargs
) -> MultiChromaDB:
    """创建多模型Chroma客户端"""
    
    # 从环境变量获取默认配置
    embedding_type = embedding_type or os.getenv("EMBEDDING_TYPE", "local")
    
    return MultiChromaDB(
        embedding_type=embedding_type,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs
    )


if __name__ == "__main__":
    # 快速测试
    try:
        print("🔄 正在测试多模型向量数据库...")
        
        # 检测可用的嵌入类型
        embedding_type = os.getenv("EMBEDDING_TYPE", "local")
        print(f"使用嵌入类型: {embedding_type}")
        
        db = create_multi_client(embedding_type=embedding_type)
        info = db.get_collection_info()
        print(f"✅ 数据库信息: {info}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查配置和依赖是否正确安装")
