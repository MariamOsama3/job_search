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

# 1. Ask user for API keys with unique widget keys
st.sidebar.title("🔑 Enter API Keys")
gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password",
    key="gemini_api_key"  # Unique key added
)
tavily_key = st.sidebar.text_input(
    "Tavily API Key", 
    type="password",
    key="tavily_api_key"  # Unique key added
)
scrapegraph_key = st.sidebar.text_input(
    "ScrapeGraph API Key", 
    type="password",
    key="scrapegraph_api_key"  # Unique key added
)

if not all([gemini_key, tavily_key, scrapegraph_key]):
    st.sidebar.error("Please provide all three API keys to continue.")
    st.stop()

try:
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_key
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini: {str(e)}")
    st.stop()

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
class SearchEngineToolInput(BaseModel):
    query: str = Field(..., description="Job search query to execute")

class SearchEngineTool(BaseTool):
    name: str = "Tavily Search"
    description: str = "Searches job postings via Tavily"
    args_schema = SearchEngineToolInput

    def _run(self, query: str):
        client = TavilyClient(api_key=tavily_key)
        return client.search(query)

class WebScrapingToolInput(BaseModel):
    url: str = Field(..., description="URL of the job posting to scrape")

class WebScrapingTool(BaseTool):
    name: str = "ScrapeGraph"
    description: str = "Extracts page details using ScrapeGraph"
    args_schema = WebScrapingToolInput

    def _run(self, url: str):
        client = Client(api_key=scrapegraph_key)
        return client.smartscraper(website_url=url)

# Main action
def start_job_search():
    # Agents
    search_rec_agent = Agent(
       role="search_recommendation_agent",
       goal="""to provide a list of recommendations search queries to be passed to the search engine.
       The queries must be varied and looking for specific items""",
       backstory="The agent is designed to help in looking for products by providing a list of suggested search queries to be passed to the search engine based on the context provided.",
       llm=llm,
       verbose=True
    )

    search_engine_agent = Agent(
        role="search engine agent",
    goal="To search on job baased on suggested search queries",
    backstory = "that egint desingned to help in finding jobs by using the suggested search queries",
        tools=[SearchEngineTool()],
        llm=llm,
        verbose=True
    )

    scraping_agent = Agent(
         role="Web scrap agent to extract information",
    goal = "to extract information from any website",
    backstory= "the egint designed to extract required information from any websie and that information will used to understand which skills the jobs need",
    
        tools=[WebScrapingTool()],
        llm=llm,
        verbose=True
    )

    analysis_agent = Agent(
       role="extract information about what requirments for every job",
    goal = "to extract information about what requirments for every job",
    backstory = "the egint should detecte what requirements for the job according to the job describtion and requirments",
        llm=llm,
        verbose=True
    )

    # Tasks
    search_task = Task(
description = "\n".join([
        "Mariam is looking for a job as {job_name}",
        "so the job must be suitable for {level}",
        "The search query must take the best offers",
        "I need links of the jobs",
        "The recommended query must not be more than {the_max_queries} ",
        "The jop must be in {country_name}"
    ]),        expected_output="List of 20 search queries",
        agent=search_rec_agent,
        output_json=SearchRecommendation
    )

    search_exec_task = Task(
        description = "\n".join([
        "search for jobs based on the suggested search queries",
        "you have to collect results from the suggested search queries",
        "ignore any results that are not related to the job",
        "Ignore any search results with confidence score less than ({score_th}) ",
        "the search result will be used to summaries the posts to understand what the candidate need to have"
        "you should give me more that 10 jop"

    ]
        expected_output="List of job postings with details",
        agent=search_engine_agent,
        output_json=AllSearchResults
    )

    scraping_task = Task(
       description = "\n".join([
        "The task is to extract job details from any job offer page url.",
        "The task has to collect results from multiple pages urls.",
        "you should focus on what requirements or qualification or responsibilities",
        "the results from you the user wil use it to understand which skills he need to have"
        "I need you to give me more than +5 jobs"
    ]
        expected_output="Structured job requirements and qualifications",
        agent=scraping_agent,
        output_json=AllExtractedProducts
    )

    analysis_task = Task(
 description = "\n".join([
        "extract what skills shoud the candidate of that job should have",
        "you have to collect results about what each job skills need",
        "ignore any results that have None values",
        "Ignore any search results with confidence score less than ({score_th}) ",
        "the candidate need to understand what skills he should have",
        "you can also recommend skills from understanding jobs title even if it not in the job description"
        "I need you to give me +10 skills"

    ]        expected_output="List of 10+ essential skills",
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

1. Install dependencies: `pip install streamlit crewai crewai-tools tavily scrapegraph_py langchain-google-genai pydantic`
2. Run: `streamlit run app.py`

Enter your API keys in the sidebar.
""")
