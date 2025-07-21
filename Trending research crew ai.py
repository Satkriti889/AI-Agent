from ddgs import DDGS
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# --- Agent base class ---
class Agent:
    def run(self, input_data):
        raise NotImplementedError

# --- Search Agent ---
class SearchAgent(Agent):
    def run(self, topic):
        print(f"Searching DuckDuckGo for trending '{topic}'...")
        with DDGS() as ddgs:
            query = f"Top trending {topic} as of July 2025"
            results = ddgs.text(query, max_results=5)
            search_texts = [r['body'] for r in results]
        return "\n".join(search_texts)

# --- Summarize Agent ---
class SummarizeAgent(Agent):
    def run(self, search_context):
        prompt = (
            f"Based on the following information about trending topics, "
            f"please provide a clear bullet-point list summarizing the top 5 trends:\n\n"
            f"{search_context}\n\n"
            f"Format the output as:\n- Trend 1\n- Trend 2\n- Trend 3\n- Trend 4\n- Trend 5"
        )
        response = llm.invoke(prompt)
        return response.content

# --- Crew (Coordinator) ---
class Crew:
    def __init__(self, agents):
        self.agents = agents

    def run(self, input_data):
        data = input_data
        for agent in self.agents:
            data = agent.run(data)
        return data

def main():
    topic = input("Enter the topic to find trending information about: ").strip()
    if not topic:
        print("Please enter a valid topic!")
        return

    crew = Crew(agents=[SearchAgent(), SummarizeAgent()])
    summary = crew.run(topic)

    print("\nSummary of top 5 trends:")
    print(summary)

if __name__ == "__main__":
    main()
