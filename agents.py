from crewai import Agent, Task, LLM
from llm_setup import basic_llm
from tools import search_engin_tool, scrap_tool

# Agent 1: Search Engine Agent
search_engine_agent = Agent(
    role="search engine agent",
    goal="To search on job baased on suggested search queries",
    backstory="that egint desingned to help in finding jobs by using the suggested search queries",
    llm=basic_llm,
    verbose=True,
    tools=[search_engin_tool]
)
search_engine_task = Task(
    description="search for jobs on websites using queries provided"
)

# Agent 2: Scrap Agent
scrap_agent = Agent(
    role="scrap agent",
    goal="To scrape job listing URLs and titles",
    backstory="This agent scrapes job listing pages",
    llm=basic_llm,
    verbose=True,
    tools=[scrap_tool]
)
scrap_task = Task(
    description="retrieve URLs and titles from job listing pages"
)

# Agent 3: Summarization Agent
summarize_agent = Agent(
    role="summarization agent",
    goal="To summarize job descriptions",
    backstory="This agent reads job descriptions and summarizes key points",
    llm=basic_llm,
    verbose=True
)
summarize_task = Task(
    description="read a job description and summarize its main responsibilities and requirements"
)

# Agent 4: Skills Extraction Agent
skills_agent = Agent(
    role="skills agent",
    goal="To extract required skills from descriptions",
    backstory="This agent extracts a list of required skills from job summaries",
    llm=basic_llm,
    verbose=True
)
skills_task = Task(
    description="analyze the job summary and output a list of required skills"
)
