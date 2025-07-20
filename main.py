import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent


load_dotenv()  # Loads the GROQ_API_KEY from your environment or .env file

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    agent_steps: List[str]
    final_answer: str

app = FastAPI()

@lru_cache(maxsize=1)
def get_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found. Put it in your .env file or as an environment variable.")

    # ***** UPDATE THIS TO A SUPPORTED MODEL *****
    llm = ChatGroq(
        model="llama3-8b-8192",  # Use a supported model from your groq account
        temperature=0,
    )

    search_tool = DuckDuckGoSearchRun()
    memory = ConversationBufferMemory(memory_key="chat_history")
    tools = [search_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    agent = get_agent()
    try:
        result = await agent.ainvoke({"input": request.question})

        # Adapt for both dict and string result types
        if isinstance(result, dict):
            agent_steps = result.get("intermediate_steps") or []
            final_answer = result.get("output") or ""
        else:
            agent_steps = []
            final_answer = str(result)

        return AskResponse(
            question=request.question,
            agent_steps=agent_steps,
            final_answer=final_answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
