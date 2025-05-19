# app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from crewai import Crew, Agent, Task, Process
from tavily import TavilyClient
from scrapegraph_py import Client
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List

# Configure Gemini AI
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = genai.GenerativeModel('gemini-1.5-flash')

# Configure Tavily
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# Configure Scrapegraph
scrape_client = Client(api_key=st.secrets["SCRAPEGRAPH_API_KEY"])

# 2. STREAMLIT APP =============================================================
st.title("Job Search Assistant")


# Define Pydantic models
class search_recommendation(BaseModel):
    search_queries: List[str] = Field(..., title="Recommended searches", min_items=1, max_items=20)

class SignleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="Page URL")
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SignleSearchResult]

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="Job page URL")
    Job_Requirements: str = Field(..., title="Job Requirements")
    Job_Title: str = Field(..., title="Job Title")
    Job_Description: str = Field(..., title="Job Description")
    Job_responsability: str = Field(..., title="Job Responsibilities")
    qualifications: str = Field(..., title="Qualifications")

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

# Streamlit UI
st.title("AI Job Search Assistant 🤖")

# User inputs
job_title = st.text_input("Job Title", "AI Developer")
level = st.selectbox("Experience Level", ["Junior", "Mid-Level", "Senior"])
country = st.text_input("Country", "Egypt")
score_th = st.slider("Confidence Score Threshold", 0.0, 1.0, 0.7)

# Initialize API clients
if st.button("Start Job Search"):
    # Set up agents and tasks
    with st.spinner("Setting up AI agents..."):
        # Initialize LLM
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        basic_llm = genai.GenerativeModel('gemini-1.5-flash')
        
        # Search Recommendation Agent
        search_recommendation_agent = Agent(
            role="Search Strategist",
            goal="Generate effective search queries for job hunting",
            backstory="Expert in crafting optimal job search strategies",
            verbose=True
        )

        # Search Engine Agent
        tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        @st.cache_data
        def search_engine_tool(query: str):
            return tavily_client.search(query)
        
        search_engine_agent = Agent(
            role="Job Researcher",
            goal="Find relevant job postings",
            backstory="Skilled in aggregating job opportunities from multiple sources",
            tools=[search_engine_tool],
            verbose=True
        )

        # Web Scraping Agent
        scrape_client = Client(api_key=st.secrets["SCRAPEGRAPH_API_KEY"])
        @st.cache_data
        def web_scraping_tool(url: str):
            return scrape_client.smartscraper(website_url=url)
        
        scraping_agent = Agent(
            role="Data Extractor",
            goal="Extract key details from job postings",
            backstory="Specializes in parsing and structuring job information",
            tools=[web_scraping_tool],
            verbose=True
        )

        # Skills Analysis Agent
        analysis_agent = Agent(
            role="Career Advisor",
            goal="Identify required skills and qualifications",
            backstory="Experienced in analyzing job requirements and career paths",
            verbose=True
        )

    # Create tasks
    with st.spinner("Processing..."):
        # Task 1: Search Recommendations
        search_task = Task(
            description=f"Generate search queries for {job_title} positions in {country} at {level} level",
            expected_output="List of 20 search queries",
            agent=search_recommendation_agent,
            output_json=search_recommendation
        )

        # Task 2: Job Search
        search_execution_task = Task(
            description=f"Find actual job postings for {job_title} in {country}",
            expected_output="List of job postings with details",
            agent=search_engine_agent,
            output_json=AllSearchResults
        )

        # Task 3: Data Extraction
        scraping_task = Task(
            description="Extract key details from job postings",
            expected_output="Structured job requirements and qualifications",
            agent=scraping_agent,
            output_json=AllExtractedProducts
        )

        # Task 4: Skills Analysis
        analysis_task = Task(
            description="Analyze required skills and qualifications",
            expected_output="List of 10+ essential skills",
            agent=analysis_agent
        )

        # Create and run crew
        job_crew = Crew(
            agents=[search_recommendation_agent, search_engine_agent, scraping_agent, analysis_agent],
            tasks=[search_task, search_execution_task, scraping_task, analysis_task],
            process=Process.sequential
        )

        results = job_crew.kickoff(inputs={
            "job_name": job_title,
            "level": level,
            "country_name": country,
            "score_th": score_th
        })

    # Display results
    st.subheader("Search Strategies")
    st.json(results['search_recommendation'])

    st.subheader("Job Opportunities")
    st.dataframe(results['search_execution'])

    st.subheader("Job Requirements Analysis")
    st.dataframe(results['scraping_task'])

    st.subheader("Essential Skills")
    st.write(results['analysis_task'])

# Deployment instructions
st.markdown("""
**To deploy:**
1. Create `secrets.toml` with:
    ```
    GEMINI_API_KEY = "your_key"
    TAVILY_API_KEY = "your_key"
    SCRAPEGRAPH_API_KEY = "your_key"
    ```
2. Run `streamlit run app.py`
""")
