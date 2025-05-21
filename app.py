import streamlit as st
import os
import json
import time
import sys
import traceback
from pydantic import BaseModel, Field
from typing import List

# Streamlit error handling for imports
st.set_page_config(
    page_title="Job Search AI Agent",
    page_icon="🔍",
    layout="wide",
)

# Function to handle missing imports and provide instructions
def import_with_error_handling():
    missing_packages = []
    
    try:
        from crewai import Crew, Agent, Task, Process, LLM
    except ImportError:
        missing_packages.append("crewai")
    
    try:
        from tavily import TavilyClient
    except ImportError:
        missing_packages.append("tavily-python")
    
    try:
        from scrapegraph_py import Client
    except ImportError:
        missing_packages.append("scrapegraph-py")
    
    try:
        from crewai.tools import tool
    except ImportError:
        if "crewai" not in missing_packages:
            missing_packages.append("crewai")
    
    try:
        import google.generativeai as genai
    except ImportError:
        missing_packages.append("google-generativeai")
        
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}", language="bash")
        st.stop()
    
    return True

# Try importing necessary libraries
if import_with_error_handling():
    from crewai import Crew, Agent, Task, Process, LLM
    from tavily import TavilyClient
    from scrapegraph_py import Client
    from crewai.tools import tool
    import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="Job Search AI Agent",
    page_icon="🔍",
    layout="wide",
)

# App title and description
st.title("🔍 Job Search AI Agent")
st.markdown("""
This application uses AI agents to help you find job listings that match your criteria.
The process involves four specialized agents working together:
1. **Search Recommendation Agent**: Suggests search queries based on your job preferences
2. **Search Engine Agent**: Searches for job listings using the recommended queries
3. **Web Scraper Agent**: Extracts detailed information from job listings
4. **Skills Summarizer Agent**: Summarizes the skills required for the job positions
""")

# Debug mode toggle
with st.sidebar:
    st.header("Debug Options")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    
    if debug_mode:
        st.write("Python version:", sys.version)
        st.write("Current directory:", os.getcwd())

# Initialize session state variables if they don't exist
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False
if "results" not in st.session_state:
    st.session_state.results = {}
if "agents_initialized" not in st.session_state:
    st.session_state.agents_initialized = False
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "running" not in st.session_state:
    st.session_state.running = False

# API Keys section
with st.expander("🔑 Configure API Keys", expanded=not st.session_state.api_keys_set):
    col1, col2 = st.columns(2)
    with col1:
        gemini_key = st.text_input("Gemini API Key", type="password", 
                             value=os.environ.get("GEMINI_API_KEY", ""))
    with col2:
        tavily_key = st.text_input("Tavily API Key", type="password",
                             value=os.environ.get("TAVILY_API_KEY", ""))
    col3, col4 = st.columns(2)
    with col3:
        scrapegraph_key = st.text_input("ScapeGraph API Key", type="password",
                                 value=os.environ.get("SCRAPEGRAPH_API_KEY", ""))
    
    if st.button("Set API Keys"):
        if gemini_key and tavily_key and scrapegraph_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            os.environ["TAVILY_API_KEY"] = tavily_key
            os.environ["SCRAPEGRAPH_API_KEY"] = scrapegraph_key
            st.session_state.api_keys_set = True
            st.success("API keys set successfully!")
        else:
            st.error("Please provide all required API keys.")

# Only show the job search form if API keys are set
if st.session_state.api_keys_set:
    # Job Search Parameters
    st.markdown("## Job Search Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        job_name = st.text_input("Job Title", "AI Developer")
        level = st.selectbox("Experience Level", ["Entry", "Mid", "Senior", "Lead", "Manager"])
    with col2:
        country_name = st.text_input("Country", "Egypt")
        score_threshold = st.slider("Minimum Score Threshold", 0.01, 1.0, 0.02, 0.01)
    
    max_queries = st.slider("Maximum Search Queries", 5, 30, 20)
    
    # Initialize and run the agents when the button is clicked
    if st.button("Start Job Search", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.show_results = False
        st.session_state.results = {}
        
        # Show a progress message
        progress_container = st.empty()
        progress_container.info("Initializing agents...")
        
        try:
            # Create output directory
            output_dir = "./ai-agent-output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize LLM
            progress_container.info("Setting up LLM...")
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            basic_llm = LLM(
                model="gemini/gemini-1.5-flash",
                temperature=0,
                provider="google_ai_studio",
                api_key=os.environ["GEMINI_API_KEY"]
            )
            
            # Define Pydantic models
            class search_recommendation(BaseModel):
                search_queries: List[str] = Field(..., title="Recommended searches to be sent to the search engines", min_items=1, max_items=max_queries)
            
            class SignleSearchResult(BaseModel):
                title: str
                url: str = Field(..., title="the page url")
                content: str
                score: float
                search_query: str

            class AllSearchResults(BaseModel):
                results: List[SignleSearchResult]
            
            class ProductSpec(BaseModel):
                specification_name: str
                specification_value: str

            class SingleExtractedProduct(BaseModel):
                page_url: str = Field(..., title="The original url of the job page")
                Job_Requirements: str = Field(..., title="The requirements of the job")
                Job_Title: str = Field(..., title="The title of the job")
                Job_Details: str = Field(title="The Details of the job", default=None)
                Job_Description: str = Field(..., title="The Description of the job")
                Job_Location: str = Field(title="The location of the job", default=None)
                Job_Salary: str = Field(title="The salary of the job", default=None)
                Job_responsability: str = Field(..., title="The responsibility of the job")
                Job_type: str = Field(title="The type of the job", default=None)
                Job_Overview: str = Field(..., title="The overview of the job")
                qualifications: str = Field(..., title="The qualifications of the job")
                product_specs: List[ProductSpec] = Field(..., title="The specifications of the product. Focus on the most important requirements.", min_items=1, max_items=5)
                agent_recommendation_notes: List[str] = Field(..., title="A set of notes why would you recommend or not recommend this job to the candidate, compared to other jobs.")

            class AllExtractedProducts(BaseModel):
                products: List[SingleExtractedProduct]
            
            # Initialize agents
            progress_container.info("Initializing Search Recommendation Agent...")
            search_recommendation_agent = Agent(
                role="search_recommendation_agent",
                goal="to provide a list of recommendations search queries to be passed to the search engine. The queries must be varied and looking for specific items",
                backstory="The agent is designed to help in looking for products by providing a list of suggested search queries to be passed to the search engine based on the context provided.",
                llm=basic_llm,
                verbose=True,
            )
            
            search_recommendation_task = Task(
                description="\n".join([
                    f"Mariam is looking for a job as {job_name}",
                    f"so the job must be suitable for {level}",
                    "The search query must take the best offers",
                    "I need links of the jobs",
                    f"The recommended query must not be more than {max_queries}",
                    f"The job must be in {country_name}"
                ]),
                expected_output="A JSON object containing a list of suggested search queries.",
                output_json=search_recommendation,
                agent=search_recommendation_agent,
                output_file=os.path.join(output_dir, "step_1_Recommend_search_queries.json"),
            )
            
            # Initialize Search Engine Agent
            progress_container.info("Initializing Search Engine Agent...")
            search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            
            @tool
            def search_engine_tool(query: str):
                """Useful for search-based queries. Use this to find current information about any query related pages using a search engine"""
                return search_client.search(query)
            
            search_engine_agent = Agent(
                role="search engine agent",
                goal="To search on job based on suggested search queries",
                backstory="This agent is designed to help in finding jobs by using the suggested search queries",
                llm=basic_llm,
                verbose=True,
                tools=[search_engine_tool]
            )
            
            search_engine_task = Task(
                description="\n".join([
                    "search for jobs based on the suggested search queries",
                    "you have to collect results from the suggested search queries",
                    "ignore any results that are not related to the job",
                    f"Ignore any search results with confidence score less than ({score_threshold})",
                    "the search result will be used to summarize the posts to understand what the candidate needs to have",
                    "you should give me more than 10 jobs"
                ]),
                expected_output="A JSON object containing search results.",
                output_json=AllSearchResults,
                agent=search_engine_agent,
                output_file=os.path.join(output_dir, "step_2_search_results.json")
            )
            
            # Initialize Web Scraper Agent
            progress_container.info("Initializing Web Scraper Agent...")
            scrape_client = Client(api_key=os.environ["SCRAPEGRAPH_API_KEY"])
            
            @tool
            def web_scraping_tool(page_url: str):
                """An AI Tool to help an agent to scrape a web page"""
                details = scrape_client.smartscraper(
                    website_url=page_url,
                    user_prompt="Extract ```json\n" + SingleExtractedProduct.schema_json() + "```\n From the web page"
                )
                return {
                    "page_url": page_url,
                    "details": details
                }
            
            search_scrap_agent = Agent(
                role="Web scrap agent to extract url information",
                goal="to extract information from any website",
                backstory="the agent designed to extract required information from any website and that information will be used to understand which skills the jobs need",
                llm=basic_llm,
                verbose=True,
                tools=[web_scraping_tool]
            )
            
            search_scrap_task = Task(
                description="\n".join([
                    "The task is to extract job details from any job offer page url.",
                    "The task has to collect results from multiple pages urls.",
                    "you should focus on what requirements or qualification or responsibilities",
                    "the results from you the user will use it to understand which skills he need to have"
                    "I need you to give me more than +5 jobs"
                ]),
                expected_output="A JSON object containing jobs details",
                output_json=AllExtractedProducts,
                output_file=os.path.join(output_dir, "step_3_search_results.json"),
                agent=search_scrap_agent
            )
            
            # Initialize Skills Summarizer Agent
            progress_container.info("Initializing Skills Summarizer Agent...")
            search_summarize_agent = Agent(
                role="extract information about what requirements for every job",
                goal="to extract information about what requirements for every job",
                backstory="the agent should detect what requirements for the job according to the job description and requirements",
                llm=basic_llm,
                verbose=True,
            )
            
            search_summarize_task = Task(
                description="\n".join([
                    "extract what skills should the candidate of that job should have",
                    "you have to collect results about what each job skills need",
                    "ignore any results that have None values",
                    f"Ignore any search results with confidence score less than ({score_threshold})",
                    "the candidate needs to understand what skills he should have",
                    "you can also recommend skills from understanding jobs title even if it not in the job description"
                    "I need you to give me +10 skills"
                ]),
                expected_output="Summary of what skills that job need candidate to have",
                agent=search_summarize_agent,
                output_file=os.path.join(output_dir, "step_4_search_results.json")
            )
            
            # Create the crew
            progress_container.info("Creating the crew and starting the job search process...")
            job_search_crew = Crew(
                process=Process.sequential,
                agents=[search_recommendation_agent, search_engine_agent, search_scrap_agent, search_summarize_agent],
                tasks=[search_recommendation_task, search_engine_task, search_scrap_task, search_summarize_task]
            )
            
            # Run the crew
            results = job_search_crew.kickoff(
                inputs={
                    "job_name": job_name,
                    "the_max_queries": max_queries,
                    "level": level,
                    "score_th": score_threshold,
                    "country_name": country_name
                }
            )
            
            # Store the results in session state
            st.session_state.results = results
            st.session_state.show_results = True
            progress_container.success("Job search completed successfully!")
            
        except Exception as e:
            progress_container.error(f"An error occurred: {str(e)}")
        
        st.session_state.running = False
        st.experimental_rerun()
    
    # Display the results if available
    if st.session_state.show_results and st.session_state.results:
        st.markdown("## Results")
        
        # Create tabs for different agent results
        tab1, tab2, tab3, tab4 = st.tabs([
            "Search Recommendations", 
            "Search Engine Results", 
            "Job Details", 
            "Skills Summary"
        ])
        
        # Try to load the output files
        try:
            # Tab 1: Search Recommendations
            with tab1:
                try:
                    with open("./ai-agent-output/step_1_Recommend_search_queries.json", "r") as f:
                        search_queries = json.load(f)
                    
                    st.subheader("Recommended Search Queries")
                    for i, query in enumerate(search_queries.get("search_queries", []), 1):
                        st.write(f"{i}. {query}")
                except FileNotFoundError:
                    if isinstance(st.session_state.results, dict) and "search_recommendation_task" in st.session_state.results:
                        st.write(st.session_state.results["search_recommendation_task"])
                    else:
                        st.info("Search recommendations not found. The search recommendation agent may still be running.")
            
            # Tab 2: Search Engine Results
            with tab2:
                try:
                    with open("./ai-agent-output/step_2_search_results.json", "r") as f:
                        search_results = json.load(f)
                    
                    st.subheader("Search Engine Results")
                    for i, result in enumerate(search_results.get("results", []), 1):
                        with st.expander(f"{i}. {result.get('title', 'Untitled')}"):
                            st.write(f"**URL:** {result.get('url', 'N/A')}")
                            st.write(f"**Score:** {result.get('score', 'N/A')}")
                            st.write(f"**Search Query:** {result.get('search_query', 'N/A')}")
                            st.write("**Content:**")
                            st.write(result.get('content', 'No content available'))
                except FileNotFoundError:
                    if isinstance(st.session_state.results, dict) and "search_engine_task" in st.session_state.results:
                        st.write(st.session_state.results["search_engine_task"])
                    else:
                        st.info("Search engine results not found. The search engine agent may still be running.")
            
            # Tab 3: Job Details
            with tab3:
                try:
                    with open("./ai-agent-output/step_3_search_results.json", "r") as f:
                        extracted_products = json.load(f)
                    
                    st.subheader("Job Details")
                    for i, product in enumerate(extracted_products.get("products", []), 1):
                        with st.expander(f"{i}. {product.get('Job_Title', 'Untitled Job')}"):
                            st.write(f"**URL:** {product.get('page_url', 'N/A')}")
                            
                            if product.get('Job_Location'):
                                st.write(f"**Location:** {product.get('Job_Location')}")
                            
                            if product.get('Job_Salary'):
                                st.write(f"**Salary:** {product.get('Job_Salary')}")
                            
                            if product.get('Job_type'):
                                st.write(f"**Job Type:** {product.get('Job_type')}")
                            
                            st.write("**Overview:**")
                            st.write(product.get('Job_Overview', 'No overview available'))
                            
                            st.write("**Requirements:**")
                            st.write(product.get('Job_Requirements', 'No requirements available'))
                            
                            st.write("**Responsibilities:**")
                            st.write(product.get('Job_responsability', 'No responsibilities available'))
                            
                            st.write("**Qualifications:**")
                            st.write(product.get('qualifications', 'No qualifications available'))
                            
                            st.write("**Key Specifications:**")
                            for spec in product.get('product_specs', []):
                                st.write(f"- **{spec.get('specification_name')}:** {spec.get('specification_value')}")
                            
                            st.write("**Agent Recommendations:**")
                            for note in product.get('agent_recommendation_notes', []):
                                st.write(f"- {note}")
                except FileNotFoundError:
                    if isinstance(st.session_state.results, dict) and "search_scrap_task" in st.session_state.results:
                        st.write(st.session_state.results["search_scrap_task"])
                    else:
                        st.info("Job details not found. The web scraper agent may still be running.")
            
            # Tab 4: Skills Summary
            with tab4:
                try:
                    with open("./ai-agent-output/step_4_search_results.json", "r") as f:
                        skills_summary = f.read()
                    
                    st.subheader("Skills Summary")
                    st.write(skills_summary)
                except FileNotFoundError:
                    if isinstance(st.session_state.results, dict) and "search_summarize_task" in st.session_state.results:
                        st.write(st.session_state.results["search_summarize_task"])
                    else:
                        st.info("Skills summary not found. The skills summarizer agent may still be running.")
        
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")

else:
    # Prompt to set API keys if not already set
    st.warning("Please set your API keys to start using the application.")
