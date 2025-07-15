"""
多模型嵌入向量数据库使用示例
支持 OpenAI、DeepSeek、Ollama、本地模型
"""

import os
from multi_chroma import create_multi_client

def test_local_model():
    """测试本地sentence-transformers模型"""
    print("=== 测试本地模型 (免费，隐私保护) ===")
    
    try:
        db = create_multi_client(
            embedding_type="local",
            collection_name="local_test"
        )
        
        # 添加测试文档
        documents = [
            "Python是一种强大的编程语言，广泛用于Web开发和数据科学",
            "机器学习是人工智能的重要分支，可以从数据中学习模式",
            "Vue.js是一个流行的JavaScript前端框架，用于构建用户界面"
        ]
        
        metadatas = [
            {"title": "Python编程介绍", "tags": ["python", "programming"], "difficulty": "beginner"},
            {"title": "机器学习概述", "tags": ["ml", "ai", "data-science"], "difficulty": "intermediate"},
            {"title": "Vue.js前端框架", "tags": ["javascript", "frontend", "vue"], "difficulty": "beginner"}
        ]
        
        db.add_documents(documents, metadatas)
        
        # 测试搜索
        results = db.smart_search("Python编程相关技术", max_results=2)
        
        print("🔍 搜索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['title']} (相似度: {result['similarity']:.3f})")
            print(f"     标签: {result['metadata'].get('tags', [])}")
        
        print("✅ 本地模型测试成功!\n")
        return True
        
    except Exception as e:
        print(f"❌ 本地模型测试失败: {e}")
        print("请安装: pip install sentence-transformers torch\n")
        return False

def test_deepseek_api():
    """测试DeepSeek API"""
    print("=== 测试DeepSeek API (便宜，中文友好) ===")
    
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("⚠️ 未设置DEEPSEEK_API_KEY，跳过测试\n")
        return False
    
    try:
        db = create_multi_client(
            embedding_type="deepseek",
            collection_name="deepseek_test",
            api_key=deepseek_key
        )
        
        documents = [
            "深度学习是机器学习的一个分支，使用多层神经网络进行学习",
            "自然语言处理是AI领域的重要分支，处理和理解人类语言"
        ]
        
        metadatas = [
            {"title": "深度学习技术", "tags": ["deep-learning", "neural-network", "ai"]},
            {"title": "自然语言处理", "tags": ["nlp", "language", "ai"]}
        ]
        
        db.add_documents(documents, metadatas)
        
        results = db.smart_search("人工智能神经网络技术", max_results=2)
        
        print("🔍 搜索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['title']} (相似度: {result['similarity']:.3f})")
        
        print("✅ DeepSeek API测试成功!\n")
        return True
        
    except Exception as e:
        print(f"❌ DeepSeek API测试失败: {e}\n")
        return False

def test_openai_api():
    """测试OpenAI API"""
    print("=== 测试OpenAI API (高质量，稍贵) ===")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ 未设置OPENAI_API_KEY，跳过测试\n")
        return False
    
    try:
        db = create_multi_client(
            embedding_type="openai",
            collection_name="openai_test",
            api_key=openai_key
        )
        
        documents = [
            "Flask是Python的轻量级Web框架，适合快速开发RESTful API",
            "FastAPI是现代Python Web框架，支持异步编程和自动API文档生成"
        ]
        
        metadatas = [
            {"title": "Flask Web框架", "tags": ["flask", "python", "web", "api"]},
            {"title": "FastAPI现代框架", "tags": ["fastapi", "python", "async", "api"]}
        ]
        
        db.add_documents(documents, metadatas)
        
        results = db.smart_search("Python Web API开发框架", max_results=2)
        
        print("🔍 搜索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['title']} (相似度: {result['similarity']:.3f})")
        
        print("✅ OpenAI API测试成功!\n")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API测试失败: {e}\n")
        return False

def comprehensive_code_example():
    """完整的代码文档示例"""
    print("=== 完整代码文档示例 ===")
    
    try:
        # 使用本地模型进行演示
        db = create_multi_client(
            embedding_type="local",
            collection_name="code_demo"
        )
        
        # 代码文档数据
        documents = [
            "这是一个Flask JWT认证中间件的实现，用于保护API端点，确保只有有效token的用户才能访问",
            "实现了完整的机器学习模型训练流程，包括数据预处理、模型训练、评估和预测",
            "React函数组件实现，用于显示用户资料信息，支持异步数据加载和错误处理"
        ]
        
        codes = [
            '''
from flask import Flask, request, jsonify
from functools import wraps
import jwt

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
        except:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated
''',
            '''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy
''',
            '''
import React, { useState, useEffect } from 'react';

const UserProfile = ({ userId }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        fetch(`/api/users/${userId}`)
            .then(response => {
                if (!response.ok) throw new Error('Failed to fetch');
                return response.json();
            })
            .then(data => {
                setUser(data);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, [userId]);
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    
    return (
        <div>
            <h2>{user.name}</h2>
            <p>{user.email}</p>
        </div>
    );
};
'''
        ]
        
        metadatas = [
            {
                "title": "Flask JWT认证中间件",
                "tags": ["python", "flask", "jwt", "security", "middleware"],
                "programming_language": "python",
                "domain": "web开发",
                "difficulty": "intermediate"
            },
            {
                "title": "机器学习模型训练",
                "tags": ["python", "ml", "scikit-learn", "data-science"],
                "programming_language": "python", 
                "domain": "数据科学",
                "difficulty": "intermediate"
            },
            {
                "title": "React用户资料组件",
                "tags": ["javascript", "react", "component", "frontend"],
                "programming_language": "javascript",
                "domain": "前端开发", 
                "difficulty": "beginner"
            }
        ]
        
        # 添加代码文档
        db.add_documents(
            documents=documents,
            metadatas=metadatas,
            codes=codes,
            ids=["code_1", "code_2", "code_3"]
        )
        
        # 测试各种查询
        queries = [
            "Python JWT身份认证怎么实现？",
            "机器学习模型训练的完整代码",
            "React组件显示用户信息",
            "Python web开发相关技术",
            "前端JavaScript组件开发"
        ]
        
        for query in queries:
            print(f"\n🔍 查询: '{query}'")
            results = db.smart_search(query, max_results=2)
            
            for i, result in enumerate(results):
                print(f"  {i+1}. 【{result['title']}】")
                print(f"     相似度: {result['similarity']:.3f}")
                print(f"     标签: {result['metadata'].get('tags', [])}")
        
        print("\n✅ 代码文档示例完成!")
        
    except Exception as e:
        print(f"❌ 代码文档示例失败: {e}")

def main():
    """主函数"""
    print("🚀 多模型嵌入向量数据库完整测试\n")
    
    # 测试各种嵌入方式
    local_success = test_local_model()
    deepseek_success = test_deepseek_api()
    openai_success = test_openai_api()
    
    # 完整代码示例
    if local_success:
        comprehensive_code_example()
    
    # 总结
    print("\n📊 测试总结:")
    print(f"  本地sentence-transformers: {'✅' if local_success else '❌'}")
    print(f"  DeepSeek API: {'✅' if deepseek_success else '⚠️ 未配置'}")
    print(f"  OpenAI API: {'✅' if openai_success else '⚠️ 未配置'}")
    
    print("\n💡 推荐使用策略:")
    print("  🎓 学习/原型开发: 本地sentence-transformers模型 (免费)")
    print("  🏢 生产环境/中文优化: DeepSeek API (便宜，中文友好)")
    print("  💼 企业级/最高质量: OpenAI API (稍贵，质量最高)")
    
    print("\n🔧 快速开始:")
    print("  1. 复制 .env.example 到 .env")
    print("  2. 根据需要配置相应的API密钥")
    print("  3. 安装依赖: pip install -r requirements_multi.txt")
    print("  4. 运行测试: python multi_example.py")


if __name__ == "__main__":
    main()
