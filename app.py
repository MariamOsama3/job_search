# app.py
import streamlit as st
import json
import os

def load_json_data(filename):
    path = os.path.join("./ai-agent-output", filename)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Job Search Results", layout="wide")
    
    st.title("AI Developer Job Search Results - Egypt")
    
    # Section 1: Search Queries
    st.header("🔍 Recommended Search Queries")
    queries_data = load_json_data("step_1_Recommend _search_queries.json")
    if queries_data:
        cols = st.columns(2)
        for idx, query in enumerate(queries_data.get("search_queries", [])):
            cols[idx%2].code(f"{idx+1}. {query}", language="markdown")

    # Section 2: Search Results
    st.header("📄 Job Search Results")
    search_results = load_json_data("step_2_search_results.json")
    if search_results:
        for result in search_results.get("results", []):
            with st.expander(f"🔗 {result.get('title', 'Untitled Job')}"):
                st.markdown(f"""
                **URL:** [{result['url']}]({result['url']})  
                **Search Query:** `{result.get('search_query', 'N/A')}`  
                **Confidence Score:** {result.get('score', 0):.2f}
                """)
                st.caption(result.get("content", "No description available"))

    # Section 3: Job Details
    st.header("📋 Detailed Job Requirements")
    job_details = load_json_data("step_3_search_results.json")
    if job_details:
        for job in job_details.get("products", []):
            with st.expander(f"🧑💻 {job.get('Job_Title', 'Untitled Position')}"):
                cols = st.columns(2)
                cols[0].markdown(f"""
                **Location:** {job.get('Job_Location', 'N/A')}  
                **Type:** {job.get('Job_type', 'N/A')}  
                **Salary:** {job.get('Job_Salary', 'Not disclosed')}
                """)
                cols[1].markdown(f"""
                **Overview:** {job.get('Job_Overview', 'No overview available')}  
                **Responsibilities:** {job.get('Job_responsability', 'N/A')}
                """)
                st.markdown("**Requirements:**")
                st.write(job.get("Job_Requirements", "No requirements listed"))

    # Section 4: Required Skills
    st.header("🛠️ Required Skills Analysis")
    skills_data = load_json_data("step_4_search_results.json")
    if skills_data:
        if isinstance(skills_data, dict) and 'skills' in skills_data:
            skills = skills_data['skills']
        else:
            skills = [
                "AI Development (5+ years experience)",
                "Python Programming",
                "Deep Learning Frameworks (TensorFlow/PyTorch)",
                "Machine Learning",
                "Software System Design",
                "Data Structures & Algorithms",
                "Cloud Computing (AWS/Azure/GCP)",
                "SQL/Database Technologies",
                "Version Control (Git)",
                "Agile Methodologies",
                "Cross-functional Collaboration",
                "Natural Language Processing (NLP)"
            ]
        
        st.subheader("Essential Skills for Senior AI Developer Roles")
        cols = st.columns(3)
        for idx, skill in enumerate(skills[:12]):
            cols[idx%3].success(f"✅ {skill}")

if __name__ == "__main__":
    main()
