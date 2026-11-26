import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

def build_knowledge_base():
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    print("   [1/2] 正在下载/加载 BGE 中文 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("   [2/2] 正在构建向量索引...")
    db = FAISS.from_documents(docs, embeddings)
    return db

vector_db = None

@app.on_event("startup")
def startup_event():
    global vector_db
    print("🚀 正在初始化 RAG 服务...")
    try:
        vector_db = build_knowledge_base()
        print("✅ 知识库加载完成！服务已就绪。")
    except Exception as e:
        print(f"❌ 知识库加载失败: {e}")

@app.post("/embed")
def get_embedding(request: QueryRequest):
    if not vector_db:
        return {"error": "Service not ready"}
    try:
        vectors = vector_db.embedding_function.embed_query(request.question)
        return {
            "text": request.question,
            "embedding": vectors,
            "dimension": len(vectors)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not vector_db:
        return {"answer": "❌ 知识库还没准备好，请稍后。"}

    llm = ChatOpenAI(
        model_name="qwen3.6-plus",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        request_timeout=120
    )

    prompt = ChatPromptTemplate.from_template("""
请根据以下上下文回答问题。如果上下文中没有相关信息，请直接说“我不知道”。

上下文:
{context}

问题:
{input}

回答:
""")
    document_chain = create_stuff_documents_chain(llm, prompt)

    # ========== 这里改成带分数的检索 ==========
    search_result = vector_db.similarity_search_with_score(request.question, k=3)
    print("===== 召回片段 + 距离分数（BGE：越小越相似） =====")
    for idx, (doc, score) in enumerate(search_result):
        print(f"【{idx+1}】score={score:.4f} 内容：{doc.page_content}")
    print("================================================")

    # 把文档拿出来传给检索链
    docs_recall = [item[0] for item in search_result]
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    try:
        response = retrieval_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"answer": f"❌ 查询出错: {str(e)}"}

if __name__ == "__main__":
    print("🌐 服务启动中: http://127.0.0.1:8000")
    print("💡 提示: 首次运行会下载 BGE 中文向量模型")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=120
    )
