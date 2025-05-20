import os
import streamlit as st
from agents import (
    search_engine_agent, search_engine_task,
    scrap_agent, scrap_task,
    summarize_agent, summarize_task,
    skills_agent, skills_task,
)

# --------------------
# Page 1: API Key Input
# --------------------
st.set_page_config(page_title="Job Search AI", layout="wide")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

page = st.sidebar.selectbox(
    "Navigation",
    ["1. API Key", "2. Job Query", "3. Results"]
)

if page == "1. API Key":
    st.title("🔑 Enter Your API Key")
    key = st.text_input("Gemini / OpenAI API Key", type="password")
    if st.button("Save Key"):
        st.session_state.api_key = key
        os.environ["GEMINI_API_KEY"] = key
        st.success("API key saved!")

# --------------------
# Page 2: Job Query
# --------------------
elif page == "2. Job Query":
    st.title("🔍 Select Job Search Term")
    query = st.text_input("Job title or keywords to search for:")
    if st.button("Run Search"):
        if not st.session_state.api_key:
            st.error("Please enter and save your API key on page 1.")
        elif not query:
            st.error("Please enter a job query to search.")
        else:
            st.session_state.query = query
            st.session_state.results = []
            # Agent 1: Generate refined search queries
            refined = search_engine_agent.run(
                task=search_engine_task,
                query=query
            )
            st.session_state.refined_queries = refined
            # Agent 2: Scrape job listings
            listings = scrap_agent.run(
                task=scrap_task,
                queries=refined
            )
            st.session_state.listings = listings
            # Agent 3: Summarize descriptions
            summaries = summarize_agent.run(
                task=summarize_task,
                listings=listings
            )
            st.session_state.summaries = summaries
            # Agent 4: Extract skills
            skills = skills_agent.run(
                task=skills_task,
                summaries=summaries
            )
            st.session_state.skills = skills
            st.success("Search completed! Go to '3. Results' to view.")

# --------------------
# Page 3: Results
# --------------------
elif page == "3. Results":
    st.title("📋 Search Results")
    if not st.session_state.get("results") and not st.session_state.get("listings"):
        st.info("No results yet. Run a search on page 2.")
    else:
        # 1) Show refined queries
        st.header("1. Search Queries Generated")
        for q in st.session_state.refined_queries:
            st.write(f"- **Query:** {q}")
        # 2) Show job listings with title and URL
        st.header("2. Job Listings Found")
        for job in st.session_state.listings:
            st.subheader(job['title'])
            st.write(f"🔗 [Job Link]({job['url']})")
        # 3) Show descriptions
        st.header("3. Job Descriptions")
        for idx, desc in enumerate(st.session_state.summaries):
            st.markdown(f"**{idx+1}.** {desc}")
        # 4) Show required skills
        st.header("4. Required Skills")
        for job_skills in st.session_state.skills:
            st.markdown(
                "- " + ", ".join(job_skills)
            )
