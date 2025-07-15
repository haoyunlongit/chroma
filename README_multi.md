# 多模型嵌入向量数据库

这是一个支持多种嵌入模型的Chroma向量数据库实现，您可以根据需求选择不同的嵌入方案。

## 🎯 支持的嵌入模型

| 模型类型 | 成本 | 质量 | 中文支持 | 隐私保护 | 适用场景 |
|---------|------|------|----------|----------|----------|
| **本地sentence-transformers** | 免费 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 学习、原型、隐私敏感 |
| **DeepSeek API** | 便宜 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中文项目、成本敏感 |
| **OpenAI API** | 较贵 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 企业级、高质量要求 |

## 🚀 快速开始

### 方法1：自动设置（推荐）
```bash
python quick_setup.py
```

### 方法2：手动设置

1. **安装依赖**
```bash
pip install -r requirements_multi.txt
```

2. **配置环境**
```bash
cp .env.example .env
# 编辑 .env 文件，设置 EMBEDDING_TYPE 和相应的API密钥
```

3. **运行测试**
```bash
python multi_example.py
```

## 📁 项目文件

- `multi_chroma.py` - 核心实现，支持多种嵌入模型
- `multi_example.py` - 使用示例和测试
- `quick_setup.py` - 自动安装和配置脚本
- `requirements_multi.txt` - 项目依赖
- `.env.example` - 环境配置示例

## 💻 基本使用

```python
from multi_chroma import create_multi_client

# 创建客户端（自动从环境变量读取配置）
db = create_multi_client()

# 添加文档
documents = ["Python是强大的编程语言", "机器学习改变世界"]
metadatas = [
    {"title": "Python教程", "tags": ["python", "programming"]},
    {"title": "ML概述", "tags": ["ml", "ai"]}
]

db.add_documents(documents, metadatas)

# 搜索文档
results = db.smart_search("Python编程", max_results=5)

for result in results:
    print(f"{result['title']}: {result['similarity']:.3f}")
```

## 🔧 配置选项

### 环境变量设置

```bash
# 选择嵌入类型
EMBEDDING_TYPE=local|deepseek|openai

# API密钥（根据选择的类型设置）
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key

# 数据库配置
CHROMA_PERSIST_DIR=./multi_chroma_db
COLLECTION_NAME=my_documents
```

### 代码中指定

```python
# 使用本地模型
db = create_multi_client(embedding_type="local")

# 使用DeepSeek API
db = create_multi_client(
    embedding_type="deepseek",
    api_key="your_deepseek_key"
)

# 使用OpenAI API
db = create_multi_client(
    embedding_type="openai",
    api_key="your_openai_key"
)
```

## 📊 特色功能

### 1. 智能代码文档处理
```python
# 添加包含代码的文档
db.add_documents(
    documents=["这是一个Flask API示例"],
    codes=["""
from flask import Flask
app = Flask(__name__)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello World'}
"""],
    metadatas=[{"title": "Flask API", "tags": ["python", "web"]}]
)
```

### 2. 智能搜索和过滤
```python
# 自动推断过滤条件
results = db.smart_search("Python web开发", max_results=5)

# 手动设置过滤条件
results = db.search(
    "机器学习", 
    where={"tags": {"$contains": "python"}}
)
```

### 3. 多维度元数据
```python
metadata = {
    "title": "深度学习教程",
    "tags": ["python", "deep-learning", "ai"],
    "difficulty": "intermediate",
    "domain": "人工智能",
    "programming_language": "python"
}
```

## 💰 成本说明

### 本地模型（免费）
- 一次性下载，永久使用
- 首次运行约需下载500MB-2GB模型
- 适合隐私敏感和成本敏感项目

### DeepSeek API（推荐）
- 约 $0.0001/1K tokens
- 100个文档约 $0.01-0.05
- 中文支持优秀，性价比最高

### OpenAI API
- 约 $0.0004/1K tokens  
- 100个文档约 $0.10-0.50
- 质量最高，生态最完善

## 🔥 最佳实践

1. **开发阶段**：使用本地模型快速迭代
2. **测试阶段**：使用DeepSeek API测试真实效果
3. **生产阶段**：根据预算选择DeepSeek或OpenAI
4. **企业内网**：使用本地模型保护数据隐私

## 🚨 常见问题

**Q: 本地模型第一次运行很慢？**
A: 正在下载模型，只需等待一次，后续运行会很快。

**Q: API调用失败？**
A: 检查API密钥是否正确，网络是否正常，配额是否充足。

**Q: 如何切换不同模型？**
A: 修改`.env`文件中的`EMBEDDING_TYPE`值即可。

**Q: 支持哪些文件格式？**
A: 支持纯文本、代码、markdown等任何可以转换为字符串的内容。

## 📞 技术支持

- 运行 `python multi_example.py` 查看完整示例
- 运行 `python quick_setup.py` 进行自动配置
- 查看代码注释了解详细用法

## 🎉 开始使用

```bash
# 一键安装配置
python quick_setup.py

# 运行示例
python multi_example.py

# 开始开发
from multi_chroma import create_multi_client
```

祝您使用愉快！🚀
