# Workaround for sqlite3 import when using pysqlite3 build
try:
    pysqlite3 = __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
from crewai import Crew, Agent, Task, Process
from crewai.tools import BaseTool
from tavily import TavilyClient
from scrapegraph_py import Client
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import GoogleGenerativeAI

# Sidebar for API keys
st.sidebar.title("🔑 Enter API Keys")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
scrapegraph_key = st.sidebar.text_input("ScrapeGraph API Key", type="password")

if not all([gemini_key, tavily_key, scrapegraph_key]):
    st.sidebar.error("Please provide all three API keys to continue.")
    st.stop()

# Initialize Gemini LLM
try:
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_key,
        request_timeout=120
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini: {str(e)}")
    st.stop()

# Pydantic Models
class SearchRecommendation(BaseModel):
    search_queries: List[str] = Field(..., min_items=3, max_items=8)

class SingleSearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult]

class SingleExtractedProduct(BaseModel):
    page_url: str
    Job_Requirements: str
    Job_Title: str
    Job_Description: str
    Job_responsibility: str
    qualifications: str

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

# App UI
st.title("AI Job Search Assistant 🤖")
col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title", "AI Developer")
    country = st.text_input("Country", "Egypt")
with col2:
    level = st.selectbox("Experience Level", ["Junior", "Mid-Level", "Senior"])
    score_th = st.slider("Confidence Score Threshold", 0.0, 1.0, 0.7)

# Tools
class SearchEngineToolInput(BaseModel):
    query: str

class SearchEngineTool(BaseTool):
    name = "Tavily Search"
    description = "Searches job postings via Tavily"
    args_schema = SearchEngineToolInput

    def _run(self, query: str):
        try:
            client = TavilyClient(api_key=tavily_key)
            result = client.search(query)
            return result if result.get('results') else {"error": "No results found"}
        except Exception as e:
            return {"error": str(e)}

class WebScrapingToolInput(BaseModel):
    url: str

class WebScrapingTool(BaseTool):
    name = "ScrapeGraph"
    description = "Extracts page details using ScrapeGraph"
    args_schema = WebScrapingToolInput

    def _run(self, url: str):
        try:
            client = Client(api_key=scrapegraph_key)
            result = client.smartscraper(website_url=url)
            return result if result.get('data') else {"error": "No data scraped"}
        except Exception as e:
            return {"error": str(e)}

# Agents

def create_agents():
    return [
        Agent(
            role="Senior Job Search Strategist",
            goal="Generate effective search queries for job platforms",
            backstory="Expert in analyzing job markets and creating targeted search strategies",
            llm=llm,
            verbose=True
        ),
        Agent(
            role="Job Search Engine Specialist",
            goal="Execute search queries and filter results",
            backstory="Skilled in using search APIs to find relevant job postings",
            tools=[SearchEngineTool()],
            llm=llm,
            verbose=True
        ),
        Agent(
            role="Web Scraping Expert",
            goal="Extract structured data from job postings",
            backstory="Specializes in parsing and extracting key information from websites",
            tools=[WebScrapingTool()],
            llm=llm,
            verbose=True
        ),
        Agent(
            role="Job Market Analyst",
            goal="Identify required skills and qualifications",
            backstory="Experienced in analyzing job requirements and market trends",
            llm=llm,
            verbose=True
        )
    ]

# Tasks

def create_tasks(job_title, level, country, score_th):
    return [
        Task(
            description=f"""
            Generate 5-8 targeted search queries for {job_title} positions in {country} 
            suitable for {level} level candidates.
            """,
            expected_output="List of search queries in JSON format",
            output_json=SearchRecommendation
        ),
        Task(
            description=f"""
            Execute the search queries and retrieve job postings.
            Only include results with a confidence score above {score_th}.
            """,
            expected_output="Structured list of job postings",
            output_json=AllSearchResults
        ),
        Task(
            description="""
            Extract job details such as requirements, qualifications, responsibilities.
            Validate the completeness of the extracted data.
            """,
            expected_output="Detailed structured job data",
            output_json=AllExtractedProducts
        ),
        Task(
            description=f"""
            Analyze job data and identify top 10+ skills and qualifications
            for {job_title} roles.
            """,
            expected_output="Markdown list of key skills"
        )
    ]

# Start job search process

def start_job_search():
    try:
        crew = Crew(
            agents=create_agents(),
            tasks=create_tasks(job_title, level, country, score_th),
            process=Process.sequential,
            memory=True,
            verbose=True
        )
        return crew.kickoff()
    except Exception as e:
        st.error(f"CrewAI process failed: {str(e)}")
        st.stop()

# UI button to start job search
if st.button("🚀 Start Job Search"):
    with st.spinner("Analyzing job market..."):
        try:
            results = start_job_search()

            st.subheader("Search Strategies")
            st.json(results[0].output.json())

            st.subheader("Top Job Opportunities")
            st.dataframe(results[1].output.results)

            st.subheader("Extracted Requirements")
            st.dataframe(results[2].output.products)

            st.subheader("Essential Skills & Qualifications")
            st.markdown(results[3].output)

        except Exception as e:
            st.error(f"Job search failed: {str(e)}")

# Deployment instructions
st.sidebar.markdown("""
**Deployment Guide:**
1. Install dependencies:  
   `pip install streamlit crewai tavily scrapegraph_py langchain-google-genai`
2. Run the app:  
   `streamlit run app.py`
3. Monitor your API key usage in each provider dashboard.
""")
