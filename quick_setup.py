#!/usr/bin/env python3
"""
多模型嵌入向量数据库快速设置脚本
自动检测环境并安装相应依赖
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("需要Python 3.8或更高版本")
        return False

def detect_embedding_preference():
    """检测用户偏好的嵌入类型"""
    print("\n🤖 选择嵌入模型类型:")
    print("1. 本地模型 (免费，隐私保护，推荐新手)")
    print("2. DeepSeek API (便宜，中文友好)")
    print("3. OpenAI API (质量最高，稍贵)")
    print("4. 全部安装 (可以随时切换)")
    
    while True:
        choice = input("\n请选择 (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        print("请输入有效选择 (1-4)")

def install_dependencies(choice):
    """根据选择安装依赖"""
    base_deps = ["chromadb>=0.4.0", "numpy>=1.21.0", "python-dotenv>=1.0.0"]
    
    if choice == '1':  # 本地模型
        deps = base_deps + ["sentence-transformers>=2.2.0", "torch>=1.13.0", "jieba>=0.42.1"]
        embedding_type = "local"
    elif choice == '2':  # DeepSeek API
        deps = base_deps + ["httpx>=0.24.0", "requests>=2.28.0"]
        embedding_type = "deepseek"
    elif choice == '3':  # OpenAI API
        deps = base_deps + ["openai>=1.0.0"]
        embedding_type = "openai"
    else:  # 全部安装
        deps = base_deps + [
            "sentence-transformers>=2.2.0", "torch>=1.13.0", "jieba>=0.42.1",
            "httpx>=0.24.0", "requests>=2.28.0", "openai>=1.0.0"
        ]
        embedding_type = "local"  # 默认使用本地模型
    
    # 安装依赖
    pip_cmd = f"pip install {' '.join(deps)}"
    success = run_command(pip_cmd, "安装Python依赖包")
    
    return success, embedding_type

def create_env_file(embedding_type):
    """创建.env文件"""
    env_content = f"""# 嵌入模型配置
EMBEDDING_TYPE={embedding_type}

# 数据库配置
CHROMA_PERSIST_DIR=./multi_chroma_db
COLLECTION_NAME=my_documents

# API密钥 (根据需要取消注释并填写)
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env 文件创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建.env文件失败: {e}")
        return False

def setup_api_keys(embedding_type):
    """设置API密钥"""
    if embedding_type == "deepseek":
        print("\n🔑 DeepSeek API设置:")
        print("1. 访问 https://platform.deepseek.com/ 注册账户")
        print("2. 获取API密钥")
        
        api_key = input("请输入DeepSeek API密钥 (回车跳过): ").strip()
        if api_key:
            # 更新.env文件
            try:
                with open('.env', 'r') as f:
                    content = f.read()
                content = content.replace('# DEEPSEEK_API_KEY=your_deepseek_api_key_here', 
                                        f'DEEPSEEK_API_KEY={api_key}')
                with open('.env', 'w') as f:
                    f.write(content)
                print("✅ DeepSeek API密钥已保存")
            except Exception as e:
                print(f"❌ 保存API密钥失败: {e}")
    
    elif embedding_type == "openai":
        print("\n🔑 OpenAI API设置:")
        print("1. 访问 https://platform.openai.com/ 注册账户")
        print("2. 获取API密钥")
        
        api_key = input("请输入OpenAI API密钥 (回车跳过): ").strip()
        if api_key:
            # 更新.env文件
            try:
                with open('.env', 'r') as f:
                    content = f.read()
                content = content.replace('# OPENAI_API_KEY=your_openai_api_key_here', 
                                        f'OPENAI_API_KEY={api_key}')
                with open('.env', 'w') as f:
                    f.write(content)
                print("✅ OpenAI API密钥已保存")
            except Exception as e:
                print(f"❌ 保存API密钥失败: {e}")

def test_installation():
    """测试安装是否成功"""
    print("\n🧪 测试安装...")
    
    test_code = """
try:
    from multi_chroma import create_multi_client
    import os
    
    # 获取嵌入类型
    embedding_type = os.getenv('EMBEDDING_TYPE', 'local')
    print(f"测试嵌入类型: {embedding_type}")
    
    # 创建客户端
    db = create_multi_client(embedding_type=embedding_type)
    info = db.get_collection_info()
    print(f"✅ 数据库连接成功: {info}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
"""
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ 安装测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 多模型嵌入向量数据库 - 快速设置")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检测用户偏好
    choice = detect_embedding_preference()
    
    # 安装依赖
    success, embedding_type = install_dependencies(choice)
    if not success:
        print("❌ 依赖安装失败，请检查网络连接和pip配置")
        return
    
    # 创建配置文件
    if not create_env_file(embedding_type):
        return
    
    # 设置API密钥
    if embedding_type in ["deepseek", "openai"]:
        setup_api_keys(embedding_type)
    
    # 测试安装
    if test_installation():
        print("\n🎉 设置完成!")
        print("\n📚 下一步:")
        print("1. 运行示例: python multi_example.py")
        print("2. 查看文档: 参考 README.md 和代码注释")
        print("3. 开始使用: from multi_chroma import create_multi_client")
        
        if embedding_type == "local":
            print("\n💡 提示: 首次运行会下载模型，可能需要几分钟")
        elif embedding_type in ["deepseek", "openai"]:
            print(f"\n💡 提示: 记得在.env文件中设置{embedding_type.upper()}_API_KEY")
    else:
        print("\n❌ 设置过程中出现问题，请检查错误信息")

if __name__ == "__main__":
    main()
