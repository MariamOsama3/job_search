# app.py
import streamlit as st
import os
import json
from typing import List
from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task, Process, LLM
from tavily import TavilyClient
from scrapegraph_py import Client
import google.generativeai as genai

# ------------ Pydantic Models (From Notebook) ------------
class search_recommendation(BaseModel):
    search_queries: List[str] = Field(..., 
                                     title="Recommended searches",
                                     min_items=1, 
                                     max_items=20)

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
    Job_Requirements: str = Field(...)
    Job_Title: str = Field(...)
    Job_Details: str = Field(default=None)
    Job_Description: str = Field(...)
    Job_Location: str = Field(default=None)
    Job_Salary: str = Field(default=None)
    Job_responsability: str = Field(...)
    Job_type: str = Field(default=None)
    Job_Overview: str = Field(...)
    qualifications: str = Field(...)
    product_specs: List[ProductSpec]
    agent_recommendation_notes: List[str]

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

# ------------ Streamlit Pages ------------
def setup_api_keys():
    st.title("🔑 API Keys Configuration")
    with st.form("api_keys"):
        gemini_key = st.text_input("Gemini API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        scrap_key = st.text_input("Scraping API Key", type="password")
        
        if st.form_submit_button("Save Keys"):
            st.session_state.gemini_key = gemini_key
            st.session_state.tavily_key = tavily_key
            st.session_state.scrap_key = scrap_key
            st.session_state.page = "search_params"
            st.rerun()

def search_parameters():
    st.title("🔍 Job Search Parameters")
    with st.form("job_search"):
        job_name = st.text_input("Job Title", "AI Developer")
        level = st.selectbox("Experience Level", ["Junior", "Mid-Level", "Senior"])
        country = st.text_input("Country", "Egypt")
        max_queries = st.slider("Max Search Queries", 5, 20, 15)
        
        if st.form_submit_button("Start Search"):
            try:
                run_crewai_pipeline(job_name, level, country, max_queries)
                st.session_state.page = "results"
                st.rerun()
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

def display_results():
    st.title("📊 Job Search Results")
    
    # Section 1: Search Queries
    with st.expander("🔎 Generated Search Queries", expanded=True):
        if 'search_queries' in st.session_state:
            for idx, query in enumerate(st.session_state.search_queries, 1):
                st.markdown(f"{idx}. `{query}`")
    
    # Section 2: Job Listings
    with st.expander("💼 Found Job Opportunities", expanded=True):
        if 'search_results' in st.session_state:
            for result in st.session_state.search_results:
                cols = st.columns([1, 4])
                with cols[0]:
                    st.metric("Confidence", f"{result.score:.0%}")
                with cols[1]:
                    st.subheader(result.title)
                    st.markdown(f"**URL:** {result.url}")
                    st.caption(result.content[:200] + "...")
    
    # Section 3: Required Skills
    with st.expander("🛠️ Required Skills Analysis", expanded=True):
        if 'skills' in st.session_state:
            for skill in st.session_state.skills:
                st.markdown(f"- {skill}")

# ------------ CrewAI Pipeline (From Notebook) ------------
def run_crewai_pipeline(job_name, level, country, max_queries):
    # Initialize clients
    genai.configure(api_key=st.session_state.gemini_key)
    tavily_client = TavilyClient(api_key=st.session_state.tavily_key)
    scrape_client = Client(api_key=st.session_state.scrap_key)

    # Define Agents (Original Prompts)
    search_agent = Agent(
        role="search_recommendation_agent",
        goal="Provide list of search queries for job search",
        backstory="Expert in generating effective search queries",
        llm=LLM(model="gemini/gemini-1.5-flash", temperature=0),
        verbose=True
    )

    # ... [Include ALL original agents/tasks from notebook] ...

    # Run pipeline
    crew = Crew(
        agents=[search_agent, engine_agent, scrap_agent, summarize_agent],
        tasks=[search_task, engine_task, scrap_task, summarize_task],
        process=Process.sequential
    )

    result = crew.kickoff(inputs={
        "job_name": job_name,
        "level": level,
        "country_name": country,
        "the_max_queries": max_queries,
        "score_th": 0.02
    })

    # Store results in session state
    st.session_state.search_queries = json.loads(result["search_queries"])
    st.session_state.search_results = json.loads(result["results"])
    st.session_state.skills = json.loads(result["skills"])

# ------------ Main App Flow ------------
if "page" not in st.session_state:
    st.session_state.page = "api_keys"

pages = {
    "api_keys": setup_api_keys,
    "search_params": search_parameters,
    "results": display_results
}

pages[st.session_state.page]()
