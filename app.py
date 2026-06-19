import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Environment Variables
# =========================
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# =========================
# LLM Models
# =========================
from langchain.chat_models import init_chat_model

llmgoogle = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai"
)

llmmodel = init_chat_model(
    "groq:llama-3.1-8b-instant"
)

# =========================
# Load Documents
# =========================
from langchain_community.document_loaders import TextLoader

cakedetail_txt = TextLoader("cakedetails.txt").load()
aboutus_txt = TextLoader("aboutus.txt").load()

# =========================
# Split Documents
# =========================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

cakedetail_doc = text_splitter.split_documents(cakedetail_txt)
aboutus_doc = text_splitter.split_documents(aboutus_txt)

# =========================
# Vector DB
# =========================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

firstdb = Chroma.from_documents(
    cakedetail_doc,
    embeddings
)

db = Chroma.from_documents(
    aboutus_doc,
    embeddings
)

detail_retriver = firstdb.as_retriever()
aboutus_retriver = db.as_retriever()

# =========================
# Tools
# =========================
from langchain_core.tools import create_retriever_tool

cakedetail_retrival_tool = create_retriever_tool(
    detail_retriver,
    "Cake_Details",
    "Tell them a valid price of cake and if the asking is not in the menu just tell we dont made that"
)

aboutus_retrival_tool = create_retriever_tool(
    aboutus_retriver,
    "About_Us",
    "Tell about our cake shop details with extraordinary manner"
)

tools = [
    aboutus_retrival_tool,
    cakedetail_retrival_tool
]

# =========================
# Agent
# =========================
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llmgoogle,
    tools=tools,
    prompt="""
    You are a cake shop assistant.
    Answer politely and professionally.
    Use the available tools whenever required.
    """
)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cake Shop Chatbot API")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):

    user_message = request.message

    result = agent.invoke(
        {
            "messages": [
                ("user", user_message)
            ]
        }
    )

    response_text = result["messages"][-1].content

    if isinstance(response_text, list):
        response_text = response_text[0]["text"]

    return {
        "response": response_text
    }