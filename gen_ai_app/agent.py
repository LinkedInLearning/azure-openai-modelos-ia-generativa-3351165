import os
import pickle
import sys

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the main directory to the Python path
sys.path.append(os.path.dirname(current_dir))

# Add the utils package to the Python path
utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(utils_dir)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Any, Union, Dict
from utils.vector_store import create_vector_store
from utils.grader import GraderUtils
from utils.graph import GraphState
from utils.generate_chain import create_generate_chain
from utils.nodes import GraphNodes
from utils.edges import EdgeGraph
from langgraph.graph import END, StateGraph
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode


load_dotenv(find_dotenv()) # important line if cannot load api key

## Getting the api keys from the .env file

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


model = ChatOpenAI(model="gpt-4o", temperature=0)
embedder = OpenAIEmbeddings()

# cargar documentos
loader = PyPDFLoader("example_data/meeting_data.pdf")
paper = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# text splits for the document
chunks_paper = text_splitter.split_documents(paper)

# storing in vector store
faiss_db_paper = FAISS.from_documents(chunks_paper, embedder)

retriever_paper = faiss_db_paper.as_retriever()

retrieval_tool = create_retriever_tool(retriever_paper, "rag_tool", "Busca información en el documento")

# Tool para buscar en Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_tool = Tool.from_function(
    func=wikipedia.run,
    name="wiki_tool",
    description="useful for when you need to search certain topic on Wikipedia, aka wiki")


tools = [wiki_tool, retrieval_tool] 

tools = [retrieval_tool, wiki_tool]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# Compile
chain = workflow.compile()

## Create the FastAPI app

app = FastAPI(
    title="Speckle Server",
    version="1.0",
    description="An API server to answer questions regarding the Speckle Developer Docs"
    
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: dict
    

# add routes
add_routes(
   app,
   chain.with_types(input_type=Input, output_type=Output),
   path="/speckle_chat",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
