import os
from crewai import Crew, Agent, Task
from crewai_tools import BaseTool
from tavily import TavilyClient
from scrapegraphai.client import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")
scrapegraph_key = os.getenv("SCRAPEGRAPH_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_key)

# Define input models
class SearchEngineToolInput(BaseModel):
    query: str = Field(..., description="Job search query to execute")

class WebScrapingToolInput(BaseModel):
    url: str = Field(..., description="URL of the job posting to scrape")

# Tool for job search using Tavily
class SearchEngineTool(BaseTool):
    name: str = "Tavily Search"
    description: str = "Searches job postings via Tavily"
    args_schema = SearchEngineToolInput

    def _run(self, query: str):
        try:
            client = TavilyClient(api_key=tavily_key)
            result = client.search(query)
            if not result.get('results'):
                raise ValueError("No results found")
            return result
        except Exception as e:
            return {"error": str(e)}

# Tool for scraping job post content using ScrapeGraph
class WebScrapingTool(BaseTool):
    name: str = "ScrapeGraph"
    description: str = "Extracts page details using ScrapeGraph"
    args_schema = WebScrapingToolInput

    def _run(self, url: str):
        try:
            client = Client(api_key=scrapegraph_key)
            result = client.smartscraper(website_url=url)
            if not result.get('data'):
                raise ValueError("No data scraped")
            return result
        except Exception as e:
            return {"error": str(e)}

# Define tools
search_tool = SearchEngineTool()
scraper_tool = WebScrapingTool()

# Define agents
job_search_agent = Agent(
    role="Job Search Expert",
    goal="Find the most relevant and recent job postings",
    backstory="An expert researcher who uses Tavily to look for jobs",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

job_scraper_agent = Agent(
    role="Job Description Extractor",
    goal="Extract detailed job information from provided URLs",
    backstory="A web scraping specialist who uses ScrapeGraph to extract content",
    tools=[scraper_tool],
    llm=llm,
    verbose=True,
)

# Define tasks
search_task = Task(
    description="Search for the latest job postings about AI research internships.",
    agent=job_search_agent,
)

scrape_task = Task(
    description="Scrape detailed job descriptions from the URLs retrieved.",
    agent=job_scraper_agent,
)

# Assemble the crew
crew = Crew(
    agents=[job_search_agent, job_scraper_agent],
    tasks=[search_task, scrape_task],
    verbose=True,
)

# Run the crew
result = crew.kickoff()
print(result)
