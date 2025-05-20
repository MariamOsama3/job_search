# Workaround for sqlite3 import when using pysqlite3 build
pysqlite3 = __import__('pysqlite3')
import sys
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from crewai import Crew, Agent, Task, Process
from crewai.tools import BaseTool
from tavily import TavilyClient
from scrapegraph_py import Client
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import GoogleGenerativeAI  # Added import

# 1. Ask user for API keys
st.sidebar.title("🔑 Enter API Keys")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
scrapegraph_key = st.sidebar.text_input("ScrapeGraph API Key", type="password")

# 2. Check inputs
if not all([gemini_key, tavily_key, scrapegraph_key]):
    st.sidebar.error("Please provide all three API keys to continue.")
    st.stop()

# 3. Configure Gemini LLM
llm = GoogleGenerativeAI(  # Changed to Langchain integration
    model="gemini-1.5-flash",
    google_api_key=gemini_key
)

# ... [Keep the Pydantic models and UI elements the same] ...

# Define CrewAI tool wrappers
class SearchEngineToolInput(BaseModel):
    query: str = Field(..., description="Job search query to execute")

class SearchEngineTool(BaseTool):
    name: str = "Tavily Search"
    description: str = "Searches job postings via Tavily"
    args_schema = SearchEngineToolInput

    def _run(self, query: str):
        client = TavilyClient(api_key=tavily_key)  # Now properly references the input key
        return client.search(query)

class WebScrapingToolInput(BaseModel):
    url: str = Field(..., description="URL of the job posting to scrape")

class WebScrapingTool(BaseTool):
    name: str = "ScrapeGraph"
    description: str = "Extracts page details using ScrapeGraph"
    args_schema = WebScrapingToolInput

    def _run(self, url: str):
        client = Client(api_key=scrapegraph_key)  # Now properly references the input key
        return client.smartscraper(website_url=url)

# Main action
def start_job_search():
    # Agents with explicit LLM configuration
    search_rec_agent = Agent(
        role="Search Strategist",
        goal="Generate effective search queries for job hunting",
        backstory="Expert in crafting optimal job search strategies",
        llm=llm,  # Added LLM configuration
        verbose=True
    )

    search_engine_agent = Agent(
        role="Job Researcher",
        goal="Find relevant job postings",
        backstory="Skilled in aggregating job opportunities from multiple sources",
        tools=[SearchEngineTool()],
        llm=llm,  # Added LLM configuration
        verbose=True
    )

    scraping_agent = Agent(
        role="Data Extractor",
        goal="Extract key details from job postings",
        backstory="Specializes in parsing and structuring job information",
        tools=[WebScrapingTool()],
        llm=llm,  # Added LLM configuration
        verbose=True
    )

    analysis_agent = Agent(
        role="Career Advisor",
        goal="Identify required skills and qualifications",
        backstory="Experienced in analyzing job requirements and career paths",
        llm=llm,  # Added LLM configuration
        verbose=True
    )
    # Tasks
    search_task = Task(
        description=f"Generate search queries for {job_title} positions in {country} at {level} level",
        expected_output="List of 20 search queries",
        agent=search_rec_agent,
        output_json=SearchRecommendation
    )

    search_exec_task = Task(
        description=f"Find actual job postings for {job_title} in {country}",
        expected_output="List of job postings with details",
        agent=search_engine_agent,
        output_json=AllSearchResults
    )

    scraping_task = Task(
        description="Extract key details from job postings",
        expected_output="Structured job requirements and qualifications",
        agent=scraping_agent,
        output_json=AllExtractedProducts
    )

    analysis_task = Task(
        description="Analyze required skills and qualifications",
        expected_output="List of 10+ essential skills",
        agent=analysis_agent
    )

    # Crew
    job_crew = Crew(
        agents=[search_rec_agent, search_engine_agent, scraping_agent, analysis_agent],
        tasks=[search_task, search_exec_task, scraping_task, analysis_task],
        process=Process.sequential
    )

    # Run
    return job_crew.kickoff(inputs={
        "job_name": job_title,
        "level": level,
        "country_name": country,
        "score_th": score_th
    })

# UI Trigger
if st.button("Start Job Search"):
    with st.spinner("Processing..."):
        results = start_job_search()

    # Display
    st.subheader("Search Strategies")
    st.json(results['search_recommendation'])

    st.subheader("Job Opportunities")
    st.dataframe(results['search_execution'])

    st.subheader("Job Requirements Analysis")
    st.dataframe(results['scraping_task'])

    st.subheader("Essential Skills")
    st.write(results['analysis_task'])

# Deployment instructions
st.sidebar.markdown("""
**To run locally:**

1. Install dependencies: `pip install streamlit crewai crewai-tools tavily scrapegraph_py google-generative-ai pydantic`
2. Run: `streamlit run app.py`

Enter your API keys in the sidebar.
""")
