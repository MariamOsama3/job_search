# app.py

import streamlit as st
import os
from agents import create_llm, create_agents, create_tasks, run_crew

st.set_page_config(page_title="AI Job Search Agents", layout="wide")

st.title("🤖 AI-Powered Job Search Assistant")

with st.sidebar:
    st.header("🔐 API Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.markdown("---")
    st.info("Enter your prompt below and run the agents.")

prompt = st.text_area("✍️ Enter your job search context, resume, or interests:", height=200)

if st.button("🚀 Run Agents") and gemini_key and prompt:
    with st.spinner("Running agents..."):
        llm = create_llm()
        agent1, agent2, agent3 = create_agents(llm)
        tasks = create_tasks(agent1, agent2, agent3, prompt)
        result = run_crew(tasks)

    st.success("✅ Agents completed their tasks!")

    # Show results nicely
    col1, col2, col3 = st.columns(3)
    col1.subheader(agent1.role)
    col1.write(tasks[0].output)

    col2.subheader(agent2.role)
    col2.write(tasks[1].output)

    col3.subheader(agent3.role)
    col3.write(tasks[2].output)

elif not gemini_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
elif not prompt:
    st.info("Enter your job-related query to begin.")
