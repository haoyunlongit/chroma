# 精简版 Chroma 向量数据库

这是一个精简版的 Chroma 向量数据库实现，提供基本的向量存储、相似性搜索和持久化功能。

## 功能特性

- ✅ **向量存储**: 支持存储文档和对应的向量嵌入
- ✅ **相似性搜索**: 基于余弦相似度的文档检索
- ✅ **元数据过滤**: 支持基于元数据的查询过滤
- ✅ **数据持久化**: 支持将数据保存到磁盘并重新加载
- ✅ **集合管理**: 支持创建和管理多个数据库集合
- ✅ **自动向量化**: 使用 TF-IDF 自动将文本转换为向量
- ✅ **自定义向量**: 支持使用预计算的向量嵌入

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基础使用

```python
from simple_chroma import create_client

# 创建客户端
client = create_client(persist_directory="./my_db")

# 创建集合
collection = client.create_collection("my_collection")

# 添加文档
documents = [
    "Python是一种编程语言",
    "机器学习是AI的分支",
    "向量数据库存储高维数据"
]

collection.add(
    documents=documents,
    metadatas=[{"type": "tech"} for _ in documents],
    ids=["doc1", "doc2", "doc3"]
)

# 查询相似文档
results = collection.query(
    query_texts=["什么是Python？"],
    n_results=2
)

print(results)
```

### 高级功能

```python
# 使用自定义向量
import numpy as np

custom_embeddings = [
    np.random.rand(100).tolist(),  # 100维向量
    np.random.rand(100).tolist()
]

collection.add(
    documents=["文档1", "文档2"],
    embeddings=custom_embeddings,
    ids=["custom1", "custom2"]
)

# 使用元数据过滤
results = collection.query(
    query_texts=["查询内容"],
    n_results=5,
    where={"type": "tech"}  # 只查询type为tech的文档
)

# 持久化数据
collection.persist()

# 删除文档
collection.delete(["doc1"])
```

## API 文档

### ChromaClient

主要的客户端类，用于管理向量数据库集合。

#### 方法

- `create_collection(name, get_or_create=False)`: 创建新集合
- `get_collection(name)`: 获取现有集合  
- `delete_collection(name)`: 删除集合
- `list_collections()`: 列出所有集合

### SimpleChroma

向量数据库集合类，提供文档存储和查询功能。

#### 主要方法

- `add(documents, embeddings=None, metadatas=None, ids=None)`: 添加文档
- `query(query_texts, n_results=10, where=None)`: 查询相似文档
- `delete(ids)`: 删除指定文档
- `count()`: 获取文档数量
- `peek(limit=10)`: 预览文档
- `persist()`: 持久化到磁盘
- `load()`: 从磁盘加载

## 运行示例

```bash
python example_usage.py
```

这将运行完整的使用示例，包括：
- 基础文档添加和查询
- 自定义向量使用
- 数据持久化演示
- 文档删除操作

## 项目结构

```
Chroma/
├── simple_chroma.py      # 核心实现
├── example_usage.py      # 使用示例
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 技术实现

- **向量化**: 使用 scikit-learn 的 TF-IDF 进行文本向量化
- **相似性计算**: 基于余弦相似度计算文档相似性
- **数据存储**: 使用 Python 字典进行内存存储
- **持久化**: 使用 pickle 进行数据序列化和反序列化

## 限制说明

这是一个精简版实现，主要用于学习和小规模应用：

- 所有数据存储在内存中，不适合大规模数据
- 向量化使用简单的 TF-IDF，不如预训练模型效果好
- 没有实现索引优化，查询速度可能较慢
- 不支持分布式部署

## 扩展建议

如需在生产环境使用，建议：

1. 集成更好的向量化模型（如 sentence-transformers）
2. 添加向量索引（如 FAISS）提升查询性能
3. 实现数据库后端存储（如 SQLite、PostgreSQL）
4. 添加更多的相似性度量方法
5. 实现批量操作优化

## 许可证

MIT License 