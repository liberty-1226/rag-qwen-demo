\# 🚀 RAG 问答系统 (基于 LangChain + Qwen)

这是一个基于 \*\*LangChain\*\*, \*\*FAISS\*\*, 和 \*\*FastAPI\*\* 构建的本地知识库检索增强生成（RAG）问答系统原型。它支持私有文档解析、向量检索与大模型精准回答，并具备 Docker 一键部署能力。

\## ✨ 主要特性
本地化部署：使用 `sentence-transformers` 进行本地向量化，保护数据隐私，零 API 成本。

高效检索：基于 FAISS 向量数据库，实现语义相似度搜索。

大模型对接：通过 OpenAI 兼容协议对接阿里云百炼（通义千问 Qwen），提供高质量回答。

工程化封装：使用 FastAPI 封装 RESTful API，支持跨语言调用；提供 Dockerfile 实现容器化部署。

质量保障：内置包含测试 Case 集，涵盖事实查询、多轮对话及边界情况，确保回答准确性。


\## 📂 项目结构
```text
rag\_project/

├── main.py              # 核心后端代码 (FastAPI + LangChain)

├── knowledge.txt        # 示例知识库文档

├── requirements.txt     # Python 依赖列表

├── Dockerfile           # Docker 镜像构建文件

├── .env.example         # 环境变量模板（请复制为 .env 并填入 Key）

├── .gitignore           # Git 忽略配置（保护 API Key）

├──test\_casas.json/ run\_tests.py      #测试套件

└── README.md            # 项目说明文档

------------------------------------------------------------------------------------------------------
# RAG 项目运行说明
## 克隆项目
```bash
git clone 仓库地址
cd rag_project


#创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

#安装依赖
pip install -r requirements.txt

#填apikey，在.env文件里
API_KEY=你的密钥

#【Hugging Face 连不上时优先执行】终端设置镜像，再启动服务
在powershell里执行（前提要先cd到你克隆的文件目录）：$env:HF_ENDPOINT="https://hf-mirror.net"

#启动服务
python main.py

#新开终端执行测试（重新打开一个powershell，同样cd到同样目录，如果是用虚拟环境需要考虑虚拟环境）
python run_tests.py
-------------------------------------------------------------------------------------------------------


