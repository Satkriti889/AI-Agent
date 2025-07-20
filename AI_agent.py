# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import os

# Load .env file
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

# Setup Together-compatible wrapper using ChatOpenAI
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_key=os.getenv('Together_API'),
    openai_api_base="https://api.together.xyz/v1",
    temperature=0.1
)

# DuckDuckGo tool
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="duckduckgo_search",
        func=search_tool.run,
        description="Use this tool to search the web for current info or facts."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.post("/ask", response_model=QueryResponse)
async def ask(query: QueryRequest):
    try:
        result = agent.run(query.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
