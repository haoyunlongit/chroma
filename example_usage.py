"""
精简版Chroma向量数据库使用示例
演示基本的向量存储、查询和持久化功能
"""

from simple_chroma import ChromaClient, create_client


def basic_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建客户端和集合
    client = create_client(persist_directory="./demo_db")
    collection = client.create_collection("demo_collection", get_or_create=True)
    
    # 添加一些示例文档
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
        "机器学习是人工智能的一个分支，通过算法让计算机学习数据模式。",
        "向量数据库专门存储和检索高维向量数据，支持相似性搜索。",
        "Chroma是一个开源的向量数据库，专为AI应用设计。",
        "自然语言处理是AI的重要领域，处理和理解人类语言。"
    ]
    
    metadatas = [
        {"category": "programming", "language": "chinese"},
        {"category": "AI", "language": "chinese"},
        {"category": "database", "language": "chinese"},
        {"category": "database", "language": "chinese"},
        {"category": "AI", "language": "chinese"}
    ]
    
    # 添加文档到数据库
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    print(f"数据库中共有 {collection.count()} 个文档")
    
    # 查看数据库内容
    print("\n--- 数据库内容预览 ---")
    preview = collection.peek(limit=3)
    for i, (doc_id, doc, meta) in enumerate(zip(preview['ids'], preview['documents'], preview['metadatas'])):
        print(f"{i+1}. ID: {doc_id}")
        print(f"   文档: {doc}")
        print(f"   元数据: {meta}")
        print()
    
    # 执行相似性查询
    print("--- 相似性查询 ---")
    query_texts = ["什么是机器学习？", "如何使用向量数据库？"]
    
    for query in query_texts:
        print(f"\n查询: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        for i, (doc_id, distance, doc, meta) in enumerate(zip(
            results['ids'][0],
            results['distances'][0], 
            results['documents'][0],
            results['metadatas'][0]
        )):
            print(f"  {i+1}. 距离: {distance:.4f}")
            print(f"     文档: {doc}")
            print(f"     元数据: {meta}")
            print()
    
    # 使用元数据过滤查询
    print("--- 元数据过滤查询 ---")
    filtered_results = collection.query(
        query_texts=["人工智能相关技术"],
        n_results=5,
        where={"category": "AI"}
    )
    
    print("只查询AI类别的文档:")
    for i, (doc_id, distance, doc) in enumerate(zip(
        filtered_results['ids'][0],
        filtered_results['distances'][0],
        filtered_results['documents'][0]
    )):
        print(f"  {i+1}. 距离: {distance:.4f} - {doc}")
    
    # 持久化数据库
    collection.persist()
    print("\n数据库已持久化保存")


def advanced_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 使用自定义向量
    client = create_client(persist_directory="./demo_db")
    collection = client.create_collection("custom_vectors", get_or_create=True)
    
    # 模拟一些预计算的向量（在实际应用中，这些可能来自预训练的模型）
    import numpy as np
    
    documents = [
        "深度学习神经网络",
        "计算机视觉图像识别", 
        "自然语言处理文本分析"
    ]
    
    # 创建一些随机向量作为示例（实际应用中应该使用真实的嵌入向量）
    custom_embeddings = [
        np.random.rand(100).tolist(),  # 100维向量
        np.random.rand(100).tolist(),
        np.random.rand(100).tolist()
    ]
    
    metadatas = [
        {"type": "deep_learning", "difficulty": "high"},
        {"type": "computer_vision", "difficulty": "medium"},
        {"type": "nlp", "difficulty": "high"}
    ]
    
    # 添加带有自定义向量的文档
    collection.add(
        documents=documents,
        embeddings=custom_embeddings,
        metadatas=metadatas,
        ids=["custom_1", "custom_2", "custom_3"]
    )
    
    print(f"添加了 {collection.count()} 个自定义向量文档")
    
    # 查询
    results = collection.query(
        query_texts=["深度学习相关内容"],
        n_results=2
    )
    
    print("\n自定义向量查询结果:")
    for i, (doc_id, distance, doc) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['documents'][0]
    )):
        print(f"  {i+1}. ID: {doc_id}, 距离: {distance:.4f}")
        print(f"     文档: {doc}")


def persistence_example():
    """持久化示例"""
    print("\n=== 持久化示例 ===")
    
    # 创建一个新的客户端，加载之前保存的数据
    client = create_client(persist_directory="./demo_db")
    
    # 列出所有可用的集合
    collections = client.list_collections()
    print(f"可用的集合: {collections}")
    
    # 加载之前的集合
    if "demo_collection" in collections:
        collection = client.get_collection("demo_collection")
        print(f"成功加载集合，包含 {collection.count()} 个文档")
        
        # 测试查询功能是否正常
        results = collection.query(
            query_texts=["数据库技术"],
            n_results=2
        )
        
        print("\n加载后的查询测试:")
        for i, (doc_id, distance, doc) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['documents'][0]
        )):
            print(f"  {i+1}. ID: {doc_id}, 距离: {distance:.4f}")
            print(f"     文档: {doc}")
    else:
        print("未找到之前保存的集合")


def cleanup_example():
    """清理示例"""
    print("\n=== 清理操作示例 ===")
    
    client = create_client(persist_directory="./demo_db")
    collection = client.get_collection("demo_collection")
    
    print(f"清理前文档数量: {collection.count()}")
    
    # 删除特定文档
    collection.delete(["doc_0", "doc_1"])
    print(f"删除后文档数量: {collection.count()}")
    
    # 查看剩余文档
    remaining = collection.peek()
    print("\n剩余文档:")
    for doc_id, doc in zip(remaining['ids'], remaining['documents']):
        print(f"  - {doc_id}: {doc}")


if __name__ == "__main__":
    try:
        # 运行基础示例
        basic_example()
        
        # 运行高级示例
        advanced_example()
        
        # 运行持久化示例
        persistence_example()
        
        # 运行清理示例
        cleanup_example()
        
        print("\n=== 所有示例运行完成 ===")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc() 