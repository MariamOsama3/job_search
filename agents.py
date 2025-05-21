# agents.py

import os
from crewai import Crew, Agent, Task, Process, LLM

def create_llm():
    return LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.7
    )

def create_agents(llm):
    # Example agent definitions
    agent_1 = Agent(
        role="Job Market Analyst",
        goal="Analyze current job market trends",
        backstory="Expert in labor economics and market trends.",
        llm=llm,
        verbose=True
    )

    agent_2 = Agent(
        role="Resume Evaluator",
        goal="Assess and improve user resume alignment",
        backstory="Experienced career coach and resume analyst.",
        llm=llm,
        verbose=True
    )

    agent_3 = Agent(
        role="Job Recommender",
        goal="Recommend best-fit jobs",
        backstory="HR specialist with access to job boards.",
        llm=llm,
        verbose=True
    )

    return agent_1, agent_2, agent_3

def create_tasks(agent_1, agent_2, agent_3, user_prompt):
    task_1 = Task(
        description=f"Analyze job market based on: {user_prompt}",
        agent=agent_1,
        expected_output="Detailed report on job trends."
    )

    task_2 = Task(
        description=f"Evaluate resume and suggest improvements: {user_prompt}",
        agent=agent_2,
        expected_output="Resume enhancement suggestions."
    )

    task_3 = Task(
        description=f"Find best-fit jobs for: {user_prompt}",
        agent=agent_3,
        expected_output="List of job recommendations."
    )

    return [task_1, task_2, task_3]

def run_crew(tasks):
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential
    )
    result = crew.kickoff()
    return result
