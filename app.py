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

# 1. Configure API keys
st.sidebar.title("🔑 Enter API Keys")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
scrapegraph_key = st.sidebar.text_input("ScrapeGraph API Key", type="password")

if not all([gemini_key, tavily_key, scrapegraph_key]):
    st.sidebar.error("Please provide all three API keys to continue.")
    st.stop()

try:
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_key,
        request_timeout=120
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini: {str(e)}")
    st.stop()

# 2. Pydantic models
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

# 3. UI Inputs
st.title("AI Job Search Assistant 🤖")
col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title", "AI Developer")
    country = st.text_input("Country", "Egypt")
with col2:
    level = st.selectbox("Experience Level", ["Junior", "Mid-Level", "Senior"])
    score_th = st.slider("Confidence Score Threshold", 0.0, 1.0, 0.7)

# 4. Tools
class SearchEngineToolInput(BaseModel):
    query: str

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
            st.error(f"Search failed: {str(e)}")
            return {"error": str(e)}

class WebScrapingToolInput(BaseModel):
    url: str

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
            st.error(f"Scraping failed: {str(e)}")
            return {"error": str(e)}

# 5. Agents
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

# 6. Tasks
def create_tasks(job_title, level, country, score_th):
    return [
        Task(
            description=f"""
            Generate 5-8 targeted search queries for {job_title} positions in {country} 
            suitable for {level} level candidates. Consider variations in job titles and
            required skills.
            """,
            expected_output="List of search queries in JSON format",
            output_json=SearchRecommendation
        ),
        Task(
            description=f"""
            Execute the provided search queries to find relevant job postings.
            Filter results with confidence score above {score_th}.
            Collect at least 10 valid job listings.
            """,
            expected_output="Structured list of job postings with metadata",
            output_json=AllSearchResults
        ),
        Task(
            description="""
            Extract detailed job requirements from the collected job postings.
            Focus on qualifications, responsibilities, and required skills.
            Validate data completeness before passing to analysis.
            """,
            expected_output="Structured job requirements data",
            output_json=AllExtractedProducts
        ),
        Task(
            description=f"""
            Analyze extracted data to identify common skills and qualifications.
            Create a list of 10+ essential requirements for {job_title} roles.
            Highlight both technical and soft skills.
            """,
            expected_output="Markdown formatted list of key skills and qualifications"
        )
    ]

# 7. Run CrewAI
def start_job_search():
    try:
        agents = create_agents()
        tasks = create_tasks(job_title, level, country, score_th)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            memory=True,
            verbose=True  # ✅ Fixed: Ensure this is a boolean
        )
        return crew.kickoff()
    except Exception as e:
        st.error(f"CrewAI process failed: {str(e)}")
        st.stop()

# 8. UI Run
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
            st.stop()

# 9. Deployment Instructions
st.sidebar.markdown("""
**Deployment Guide:**
1. Install requirements:  
   `pip install streamlit crewai tavily scrapegraph_py langchain-google-genai`
2. Run with:  
   `streamlit run app.py`
3. Monitor API usage in provider dashboards
""")
