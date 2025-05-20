# app.py
import os
import streamlit as st
from pydantic import BaseModel, Field
from typing import List
from crewai import Crew, Agent, Task, Process
from crewai import LLM
import google.generativeai as genai
from tavily import TavilyClient
from scrapegraph_py import Client

# Disable AgentOps tracking
os.environ["AGENTOPS_API_KEY"] = "dummy-key"

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

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="Job page URL")
    Job_Requirements: str = Field(..., title="Job requirements")
    Job_Title: str = Field(..., title="Job title")
    Job_Details: str = Field(title="Job details", default=None)
    Job_Description: str = Field(..., title="Job description")
    Job_Location: str = Field(title="Job location", default=None)
    Job_Salary: str = Field(title="Job salary", default=None)
    Job_responsability: str = Field(..., title="Job responsibilities")
    Job_type: str = Field(title="Job type", default=None)
    Job_Overview: str = Field(..., title="Job overview")
    qualifications: str = Field(..., title="Qualifications")
    product_specs: List[ProductSpec] = Field(..., title="Key specifications", min_items=1, max_items=5)
    agent_recommendation_notes: List[str] = Field(..., title="Recommendation notes")

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

def validate_gemini_key(key: str) -> bool:
    """Validate Gemini API key with detailed error handling"""
    try:
        genai.configure(api_key=key)
        models = genai.list_models()
        if not any('gemini' in model.name for model in models):
            st.error("Gemini models not available with this API key")
            return False
        return True
    except Exception as e:
        st.error(f"""
            🔴 Critical Gemini API Error: {str(e)}
            
            Required Steps:
            1. Create API key at https://aistudio.google.com/app/apikey
            2. Enable billing: https://console.cloud.google.com/billing
            3. Enable Generative Language API: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
            4. Check key for extra spaces or missing characters
            """)
        return False

def validate_tavily_key(key: str) -> bool:
    """Validate Tavily API key"""
    try:
        client = TavilyClient(api_key=key)
        client.search("test query")
        return True
    except Exception as e:
        st.error(f"""
            🔴 Tavily API Error: {str(e)}
            Get your free API key at https://app.tavily.com
            """)
        return False

def main():
    st.title("AI Job Search Assistant")
    
    # Session state management
    if 'api_keys_valid' not in st.session_state:
        st.session_state.api_keys_valid = False

    # API Key Validation Section
    if not st.session_state.api_keys_valid:
        with st.form("api_keys"):
            st.header("🔑 API Key Configuration")
            gemini_key = st.text_input("Gemini API Key", type="password", 
                                     help="Get from https://aistudio.google.com/app/apikey")
            tavily_key = st.text_input("Tavily API Key", type="password",
                                     help="Get from https://app.tavily.com")
            scrapegraph_key = st.text_input("Scrapegraph API Key", type="password")
            
            if st.form_submit_button("🔒 Validate & Save API Keys"):
                if all([gemini_key, tavily_key, scrapegraph_key]):
                    if validate_gemini_key(gemini_key) and validate_tavily_key(tavily_key):
                        st.session_state.update({
                            'gemini_key': gemini_key.strip(),
                            'tavily_key': tavily_key.strip(),
                            'scrapegraph_key': scrapegraph_key.strip(),
                            'api_keys_valid': True
                        })
                        st.success("✅ API Keys Validated Successfully!")
                        st.rerun()
                else:
                    st.error("All API keys are required")

    if st.session_state.api_keys_valid:
        # Job Search Parameters
        with st.form("job_params"):
            st.header("🔍 Job Search Parameters")
            job_name = st.text_input("Job Title", "AI Developer")
            level = st.selectbox("Experience Level", ["Junior", "Mid", "Senior"])
            country_name = st.text_input("Country", "Egypt")
            score_th = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            
            if st.form_submit_button("🚀 Start Job Search"):
                try:
                    # Initialize APIs with fresh validation
                    if not validate_gemini_key(st.session_state.gemini_key):
                        st.stop()
                        
                    genai.configure(api_key=st.session_state.gemini_key)
                    tavily_client = TavilyClient(api_key=st.session_state.tavily_key)
                    scrape_client = Client(api_key=st.session_state.scrapegraph_key)

                    # Initialize LLM with double verification
                    basic_llm = LLM(
                        model="gemini/gemini-1.5-flash",
                        temperature=0,
                        provider="google_ai_studio",
                        api_key=st.session_state.gemini_key,
                        config={"max_retries": 3}
                    )

                    # Create Agents
                    search_agent = Agent(
                        role="Search Recommendation Agent",
                        goal="Generate precise search queries",
                        backstory="Expert in creating optimized job search queries",
                        llm=basic_llm,
                        verbose=True
                    )

                    search_engine_agent = Agent(
                        role="Search Engine Agent",
                        goal="Execute targeted job searches",
                        backstory="Specializes in multi-platform job search execution",
                        llm=basic_llm,
                        verbose=True
                    )

                    scrap_agent = Agent(
                        role="Web Scraping Agent",
                        goal="Extract detailed job information",
                        backstory="Expert in parsing and structuring job data",
                        llm=basic_llm,
                        verbose=True
                    )

                    # Create Tasks with enhanced validation
                    search_task = Task(
                        description=f"""
                        Generate search queries for {job_name} positions in {country_name}
                        - Experience level: {level}
                        - Include major job platforms
                        - Max 20 queries
                        """,
                        expected_output="Structured list of search queries",
                        output_json=SearchRecommendation,
                        agent=search_agent
                    )

                    search_engine_task = Task(
                        description=f"""
                        Execute job searches with parameters:
                        - Minimum confidence score: {score_th}
                        - Collect 10-20 relevant positions
                        - Filter out low-quality results
                        """,
                        expected_output="Validated job listings",
                        output_json=AllSearchResults,
                        agent=search_engine_agent
                    )

                    scrap_task = Task(
                        description="""
                        Extract detailed job specifications:
                        - Requirements and qualifications
                        - Key responsibilities
                        - Salary ranges and benefits
                        """,
                        expected_output="Structured job profiles",
                        output_json=AllExtractedProducts,
                        agent=scrap_agent
                    )

                    # Create and run crew with error handling
                    job_crew = Crew(
                        agents=[search_agent, search_engine_agent, scrap_agent],
                        tasks=[search_task, search_engine_task, scrap_task],
                        process=Process.sequential,
                        memory=True
                    )

                    # Execute pipeline with progress tracking
                    with st.spinner("🔍 Scanning job market..."):
                        results = job_crew.kickoff(inputs={
                            "job_name": job_name,
                            "level": level,
                            "country_name": country_name,
                            "score_th": score_th
                        })

                    # Display results with clear sections
                    st.header("📊 Job Search Results")
                    
                    with st.expander("🔎 Recommended Search Queries", expanded=True):
                        st.json(results.get('search_recommendation', {}))
                    
                    with st.expander("📋 Job Listings"):
                        st.json(results.get('search_results', {}))
                    
                    with st.expander("📄 Detailed Job Analysis"):
                        st.json(results.get('job_details', {}))

                except Exception as e:
                    st.error(f"""
                    🚨 Critical Error: {str(e)}
                    
                    Troubleshooting Steps:
                    1. Verify API keys are still valid
                    2. Check billing status for Gemini API
                    3. Try simpler search parameters
                    4. Contact support if issue persists
                    """)
                    st.stop()

        # Reset API keys
        if st.button("🔄 Reset API Keys"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
