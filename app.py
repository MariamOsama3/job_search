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

def main():
    st.title("AI Job Search Assistant")
    
    # Session state management
    if 'api_keys_entered' not in st.session_state:
        st.session_state.api_keys_entered = False

    # API Key Input Section
    if not st.session_state.api_keys_entered:
        with st.form("api_keys"):
            st.header("API Key Configuration")
            gemini_key = st.text_input("Gemini API Key", type="password")
            tavily_key = st.text_input("Tavily API Key", type="password")
            scrapegraph_key = st.text_input("Scrapegraph API Key", type="password")
            
            if st.form_submit_button("Save API Keys"):
                if gemini_key and tavily_key and scrapegraph_key:
                    st.session_state.update({
                        'gemini_key': gemini_key,
                        'tavily_key': tavily_key,
                        'scrapegraph_key': scrapegraph_key,
                        'api_keys_entered': True
                    })
                    st.rerun()
                else:
                    st.error("Please fill all API key fields")

    if st.session_state.api_keys_entered:
        # Job Search Parameters
        with st.form("job_params"):
            st.header("Job Search Parameters")
            job_name = st.text_input("Job Title", "AI Developer")
            level = st.selectbox("Experience Level", ["Junior", "Mid", "Senior"])
            country_name = st.text_input("Country", "Egypt")
            score_th = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            
            if st.form_submit_button("Start Search"):
                try:
                    # Initialize APIs
                    genai.configure(api_key=st.session_state.gemini_key)
                    tavily_client = TavilyClient(api_key=st.session_state.tavily_key)
                    scrape_client = Client(api_key=st.session_state.scrapegraph_key)

                    # Initialize LLM
                    basic_llm = LLM(
                        model="gemini/gemini-1.5-flash",
                        temperature=0,
                        provider="google_ai_studio",
                        api_key=st.session_state.gemini_key
                    )

                    # Create Agents
                    search_agent = Agent(
                        role="Search Recommendation Agent",
                        goal="Generate search queries for job search",
                        backstory="Expert in creating effective search queries for job platforms",
                        llm=basic_llm,
                        verbose=True
                    )

                    search_engine_agent = Agent(
                        role="Search Engine Agent",
                        goal="Search for jobs based on recommended queries",
                        backstory="Specialized in executing job searches using various platforms",
                        llm=basic_llm,
                        verbose=True,
                        tools=[lambda query: tavily_client.search(query)]
                    )

                    scrap_agent = Agent(
                        role="Web Scraping Agent",
                        goal="Extract job details from URLs",
                        backstory="Expert in scraping and parsing job postings",
                        llm=basic_llm,
                        verbose=True,
                        tools=[lambda url: scrape_client.smartscraper(url)]
                    )

                    # Create Tasks
                    search_task = Task(
                        description="\n".join([
                            f"Generate search queries for {job_name} positions",
                            f"Target experience level: {level}",
                            f"Location: {country_name}",
                            "Include major job platforms and company websites"
                        ]),
                        expected_output="List of search queries",
                        output_json=SearchRecommendation,
                        agent=search_agent
                    )

                    search_engine_task = Task(
                        description="\n".join([
                            "Execute searches on job platforms",
                            f"Minimum confidence score: {score_th}",
                            "Collect at least 10 relevant job postings",
                            "Filter out irrelevant results"
                        ]),
                        expected_output="Structured search results",
                        output_json=AllSearchResults,
                        agent=search_engine_agent
                    )

                    scrap_task = Task(
                        description="\n".join([
                            "Extract detailed information from job postings",
                            "Focus on requirements and qualifications",
                            "Identify key skills and specifications"
                        ]),
                        expected_output="Structured job details",
                        output_json=AllExtractedProducts,
                        agent=scrap_agent
                    )

                    # Create and run crew
                    job_crew = Crew(
                        agents=[search_agent, search_engine_agent, scrap_agent],
                        tasks=[search_task, search_engine_task, scrap_task],
                        process=Process.sequential
                    )

                    # Execute pipeline
                    with st.spinner("Processing job search..."):
                        results = job_crew.kickoff(inputs={
                            "job_name": job_name,
                            "level": level,
                            "country_name": country_name,
                            "score_th": score_th
                        })

                    # Display results
                    st.header("Job Search Results")
                    
                    with st.expander("Recommended Search Queries"):
                        st.json(results['search_recommendation'])
                    
                    with st.expander("Job Listings"):
                        st.json(results['search_results'])
                    
                    with st.expander("Detailed Job Analysis"):
                        st.json(results['job_details'])

                except Exception as e:
                    st.error(f"Error in job search pipeline: {str(e)}")

        # Reset API keys
        if st.button("Reset API Keys"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
