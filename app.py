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

# 1. Ask user for API keys
st.sidebar.title("🔑 Enter API Keys")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
scrapegraph_key = st.sidebar.text_input("ScrapeGraph API Key", type="password")

# 2. Check inputs
if not all([gemini_key, tavily_key, scrapegraph_key]):
    st.sidebar.error("Please provide all three API keys to continue.")
    st.stop()

# 3. Configure clients
# Gemini AI
genai.configure(api_key=gemini_key)
llm = genai.GenerativeModel('gemini-1.5-flash')

# Tavily and ScrapeGraph clients will be wrapped in tools

# 4. Streamlit App UI
st.title("AI Job Search Assistant 🤖")

# Define Pydantic models
class SearchRecommendation(BaseModel):
    search_queries: List[str] = Field(..., title="Recommended searches", min_items=1, max_items=20)

class SingleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="Page URL")
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult]

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="Job page URL")
    Job_Requirements: str = Field(..., title="Job Requirements")
    Job_Title: str = Field(..., title="Job Title")
    Job_Description: str = Field(..., title="Job Description")
    Job_responsibility: str = Field(..., title="Job Responsibilities")
    qualifications: str = Field(..., title="Qualifications")

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

# User inputs
job_title = st.text_input("Job Title", "AI Developer")
level = st.selectbox("Experience Level", ["Junior", "Mid-Level", "Senior"])
country = st.text_input("Country", "Egypt")
score_th = st.slider("Confidence Score Threshold", 0.0, 1.0, 0.7)

# Define CrewAI tool wrappers
class SearchEngineTool(BaseTool):
    name: str = "Tavily Search"
    description: str = "Searches job postings via Tavily"

    def run(self, query: str):
        client = TavilyClient(api_key=tavily_key)
        return client.search(query)

class WebScrapingTool(BaseTool):
    name: str = "ScrapeGraph"
    description: str = "Extracts page details using ScrapeGraph"

    def run(self, url: str):
        client = Client(api_key=scrapegraph_key)
        return client.smartscraper(website_url=url)

# Main action
def start_job_search():
    # Ensure Gemini is configured
    genai.configure(api_key=gemini_key)
    basic_llm = genai.GenerativeModel('gemini-1.5-flash')

    # Agents
    search_rec_agent = Agent(
        role="Search Strategist",
        goal="Generate effective search queries for job hunting",
        backstory="Expert in crafting optimal job search strategies",
        verbose=True
    )

    search_engine_agent = Agent(
        role="Job Researcher",
        goal="Find relevant job postings",
        backstory="Skilled in aggregating job opportunities from multiple sources",
        tools=[SearchEngineTool()],
        verbose=True
    )

    scraping_agent = Agent(
        role="Data Extractor",
        goal="Extract key details from job postings",
        backstory="Specializes in parsing and structuring job information",
        tools=[WebScrapingTool()],
        verbose=True
    )

    analysis_agent = Agent(
        role="Career Advisor",
        goal="Identify required skills and qualifications",
        backstory="Experienced in analyzing job requirements and career paths",
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
