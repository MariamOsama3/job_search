# app.py
import streamlit as st
import os
import json
from pydantic import BaseModel, Field
from typing import List
from crewai import Crew, Agent, Task, Process
import google.generativeai as genai
from tavily import TavilyClient
from scrapegraph_py import Client

# ---------------------------
# Streamlit UI Configuration
# ---------------------------
st.set_page_config(page_title="AI Job Search Assistant", layout="wide")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# ---------------------------
# Pydantic Models
# ---------------------------
class SearchRecommendation(BaseModel):
    search_queries: List[str] = Field(..., 
                    title="Recommended searches",
                    min_items=1, 
                    max_items=20)

class SingleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="Page URL")
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult]

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="Job page URL")
    Job_Requirements: str = Field(..., title="Job requirements")
    Job_Title: str = Field(..., title="Job title")
    Job_Details: str = Field(title="Job details", default=None)
    Job_Description: str = Field(..., title="Job description")
    Job_Location: str = Field(title="Location", default=None)
    Job_Salary: str = Field(title="Salary", default=None)
    Job_responsability: str = Field(..., title="Responsibilities")
    Job_type: str = Field(title="Job type", default=None)
    Job_Overview: str = Field(..., title="Job overview")
    qualifications: str = Field(..., title="Qualifications")
    product_specs: List[ProductSpec] = Field(...,
                    title="Key specifications",
                    min_items=1,
                    max_items=5)
    agent_recommendation_notes: List[str] = Field(...,
                        title="Recommendation notes")

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

# ---------------------------
# Streamlit Components
# ---------------------------
def show_sidebar():
    with st.sidebar:
        st.header("🔑 API Configuration")
        gemini_key = st.text_input("GEMINI API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        scrap_key = st.text_input("ScrapeGraph API Key", type="password")
        
        st.header("⚙️ Search Parameters")
        the_max_queries = st.slider("Max Search Queries", 5, 20, 10)
        score_th = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        
        return {
            "gemini": gemini_key,
            "tavily": tavily_key,
            "scrap": scrap_key,
            "max_queries": the_max_queries,
            "score_th": score_th
        }

def main_form(config):
    with st.form("job_search_form"):
        st.header("🔍 Job Search Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            job_name = st.text_input("Job Title", "AI Developer")
            country_name = st.text_input("Country", "Egypt")
            
        with col2:
            level = st.selectbox("Experience Level", 
                               ["Junior", "Mid-Level", "Senior"])
        
        if st.form_submit_button("🚀 Start Job Search"):
            if not all(config.values()):
                st.error("Please provide all API keys!")
                return None
            
            return {
                "job_name": job_name,
                "level": level,
                "country_name": country_name,
                **config
            }
    return None

# ---------------------------
# AI Components
# ---------------------------
def initialize_agents(config):
    # Initialize AI models
    genai.configure(api_key=config['gemini'])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Initialize clients
    search_client = TavilyClient(api_key=config['tavily'])
    scrape_client = Client(api_key=config['scrap'])
    
    # Define tools
    @tool
    def search_engine_tool(query: str):
        return search_client.search(query)
    
    @tool
    def web_scraping_tool(page_url: str):
        details = scrape_client.smartscraper(
            website_url=page_url,
            user_prompt="Extract job details from page"
        )
        return {"page_url": page_url, "details": details}
    
    # Create agents
    agents = {
        "search_recommendation": Agent(
            role="Search Query Generator",
            goal="Create effective job search queries",
            backstory="Expert in creating targeted search queries",
            llm=model,
            verbose=True
        ),
        "search_engine": Agent(
            role="Job Search Agent",
            goal="Find relevant job postings",
            backstory="Skilled in job market research",
            llm=model,
            verbose=True,
            tools=[search_engine_tool]
        ),
        "scrap_agent": Agent(
            role="Web Scraping Agent",
            goal="Extract job details from websites",
            backstory="Expert in web data extraction",
            llm=model,
            verbose=True,
            tools=[web_scraping_tool]
        ),
        "summarize_agent": Agent(
            role="Skills Analyst",
            goal="Identify required skills",
            backstory="Expert in analyzing job requirements",
            llm=model,
            verbose=True
        )
    }
    
    return agents

def create_tasks(input_params):
    return [
        Task(
            description=f"""
            Generate search queries for {input_params['job_name']} positions
            in {input_params['country_name']} for {input_params['level']} level
            """,
            expected_output="List of search queries",
            output_json=SearchRecommendation
        ),
        Task(
            description=f"""
            Find job postings with confidence above {input_params['score_th']}
            """,
            expected_output="Job search results",
            output_json=AllSearchResults
        ),
        Task(
            description="Extract details from job postings",
            expected_output="Detailed job information",
            output_json=AllExtractedProducts
        ),
        Task(
            description="Analyze required skills and qualifications",
            expected_output="List of required skills"
        )
    ]

# ---------------------------
# Main Execution
# ---------------------------
def main():
    st.title("🤖 AI-Powered Job Search Assistant")
    
    # Get configuration
    config = show_sidebar()
    input_params = main_form(config)
    
    if input_params:
        with st.spinner("🔍 Analyzing job market..."):
            try:
                # Initialize AI components
                agents = initialize_agents(config)
                tasks = create_tasks(input_params)
                
                # Create and run crew
                crew = Crew(
                    agents=list(agents.values()),
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=2
                )
                
                # Store results in session state
                st.session_state.results = crew.kickoff(inputs={
                    "job_name": input_params['job_name'],
                    "the_max_queries": input_params['max_queries'],
                    "level": input_params['level'],
                    "score_th": input_params['score_th'],
                    "country_name": input_params['country_name']
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

        # Display results
        if st.session_state.results:
            st.success("✅ Analysis Complete!")
            
            with st.expander("🔎 Generated Search Queries", expanded=True):
                st.json(st.session_state.results.get('search_queries', []))
            
            with st.expander("📄 Job Search Results"):
                st.json(st.session_state.results.get('search_results', []))
            
            with st.expander("📋 Extracted Job Details"):
                st.json(st.session_state.results.get('scraped_data', []))
            
            with st.expander("📚 Required Skills Analysis"):
                st.write(st.session_state.results.get('skills_summary', ""))

if __name__ == "__main__":
    main()
