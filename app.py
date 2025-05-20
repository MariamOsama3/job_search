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

def main():
    st.title("AI Job Search Assistant")
    
    # Initialize session state
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
                    st.session_state.gemini_key = gemini_key
                    st.session_state.tavily_key = tavily_key
                    st.session_state.scrapegraph_key = scrapegraph_key
                    st.session_state.api_keys_entered = True
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
                    # Initialize APIs with user-provided keys
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

                    # Create agents and tasks
                    search_agent = Agent(
                        role="Search Recommendation Agent",
                        goal="Generate search queries for job search",
                        backstory="Expert in creating effective search queries for job platforms",
                        llm=basic_llm,
                        verbose=True
                    )

                    search_task = Task(
                        description=f"Generate search queries for {job_name} positions in {country_name} for {level} level",
                        expected_output="JSON list of search queries",
                        output_json=SearchRecommendation,
                        agent=search_agent
                    )

                    # Create and run crew
                    job_crew = Crew(
                        agents=[search_agent],
                        tasks=[search_task],
                        process=Process.sequential
                    )

                    # Execute crew
                    with st.spinner("Searching for jobs..."):
                        results = job_crew.kickoff(inputs={
                            "job_name": job_name,
                            "level": level,
                            "country_name": country_name
                        })

                    # Display results
                    st.header("Search Recommendations")
                    st.json(results)

                except Exception as e:
                    st.error(f"Error in job search pipeline: {str(e)}")

        # Add button to reset API keys
        if st.button("Reset API Keys"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
