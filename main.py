import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 使用本地免费的 HuggingFace 模型做 Embedding，避免阿里云 API 兼容性问题
from langchain_community.embeddings import HuggingFaceEmbeddings
# Chat 模型依然使用阿里云百炼（通义千问）
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. 加载 .env 文件中的密码
load_dotenv()

# 2. 初始化 FastAPI (这是一个网站框架，用来接收用户的提问)
app = FastAPI()

# 3. 定义用户提问的格式 (JSON 格式)
class QueryRequest(BaseModel):
    question: str

# 4. 核心函数：构建知识库 (把 txt 变成向量数据库)
def build_knowledge_base():
    # 4.1 读取 knowledge.txt
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()
    
    # 4.2 切分文本 (把长文章切成每段 100 字的小块，方便检索)
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # 4.3 向量化 (使用本地免费模型，无需 API Key，速度快且稳定)
    print("   [1/2] 正在下载/加载本地 Embedding 模型 (约 100MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" 
    )
    
    print("   [2/2] 正在构建向量索引...")
    db = FAISS.from_documents(docs, embeddings)
    return db

# 全局变量，用来存建好的数据库
vector_db = None

# 5. 启动服务时，自动加载知识库
@app.on_event("startup")
def startup_event():
    global vector_db
    print("🚀 正在初始化 RAG 服务...")
    try:
        vector_db = build_knowledge_base()
        print("✅ 知识库加载完成！服务已就绪。")
    except Exception as e:
        print(f"❌ 知识库加载失败: {e}")

# 6. 【新增】通用 Embedding 接口 (供其他系统调用)
@app.post("/embed")
def get_embedding(request: QueryRequest):
    """
    这个接口可以被任何系统调用，用来获取文字的向量表示
    """
    if not vector_db:
        return {"error": "Service not ready"}
    
    try:
        # 计算向量
        vectors = vector_db.embedding_function.embed_query(request.question)
        return {
            "text": request.question,
            "embedding": vectors,
            "dimension": len(vectors)
        }
    except Exception as e:
        return {"error": str(e)}

# 7. 定义问答接口：当用户访问 /ask 时，执行这个函数
@app.post("/ask")
def ask_question(request: QueryRequest):
    if not vector_db:
        return {"answer": "❌ 知识库还没准备好，请稍后。"}
    
    # 7.1 初始化大模型 (使用阿里云百炼的通义千问)
    llm = ChatOpenAI(
        model_name="qwen3.6-plus", # 阿里云通义千问 Turbo 模型
        temperature=0, # 温度越低，回答越严谨
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL") # 确保 .env 里有阿里云的 Base URL
    )
    
    # 7.2 定义提示词模板 (告诉 AI 怎么回答问题)
    prompt = ChatPromptTemplate.from_template("""
    请根据以下上下文回答问题。如果上下文中没有相关信息，请直接说“我不知道”。
    
    上下文:
    {context}
    
    问题:
    {input}
    
    回答:
    """)
    
    # 7.3 创建文档处理链
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 7.4 创建检索器
    retriever = vector_db.as_retriever()
    
    # 7.5 组装完整的 RAG 链条
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 7.6 执行查询
    try:
        response = retrieval_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"answer": f"❌ 查询出错: {str(e)}"}

# 8. 启动服务器
if __name__ == "__main__":
    print("🌐 服务启动中: http://127.0.0.1:8000")
    print("💡 提示: 首次运行会下载 Embedding 模型，请耐心等待...")
    uvicorn.run(app, host="0.0.0.0", port=8000)