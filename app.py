# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from crewai import Crew, Agent, Task, Process
from crewai import LLM
import google.generativeai as genai
from tavily import TavilyClient
from scrapegraph_py import Client

# Load environment variables
load_dotenv()

# Initialize APIs
def initialize_apis():
    try:
        # Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Tavily
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Scrapegraph
        scrape_client = Client(api_key=os.getenv("SCRAPEGRAPH_API_KEY"))
        
        return model, tavily_client, scrape_client
    except Exception as e:
        st.error(f"Error initializing APIs: {str(e)}")
        return None, None, None

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

# Streamlit UI
def main():
    st.title("AI Job Search Assistant")
    
    # Input Section
    with st.form("job_params"):
        st.header("Job Search Parameters")
        job_name = st.text_input("Job Title", "AI Developer")
        level = st.selectbox("Experience Level", ["Junior", "Mid", "Senior"])
        country_name = st.text_input("Country", "Egypt")
        score_th = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        submit_button = st.form_submit_button("Start Search")

    if submit_button:
        # Initialize agents and tasks
        try:
            # Initialize LLM
            basic_llm = LLM(
                model="gemini/gemini-1.5-flash",
                temperature=0,
                provider="google_ai_studio",
                api_key=os.getenv("GEMINI_API_KEY")
            )

            # Create agents
            search_agent = Agent(
                role="Search Recommendation Agent",
                goal="Generate search queries for job search",
                backstory="Expert in creating effective search queries for job platforms",
                llm=basic_llm,
                verbose=True
            )

            # Create tasks
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

if __name__ == "__main__":
    main()
