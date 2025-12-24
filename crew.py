from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import json
import ast
import os

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    provider="openai"  # Treat as OpenAI-compatible
)


@CrewBase
class AnalyzeRequestCrew:
    """Analyze Request crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def senior_story_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['senior_story_analyst'],
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

    @task
    def analyze_request(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_request_task'],
            agent=self.senior_story_analyst_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


@CrewBase
class GenerateStoryCrew:
    """Generate Story crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def senior_story_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['senior_story_writer'],
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

    @task
    def generate_story(self) -> Task:
        return Task(
            config=self.tasks_config['generate_story_task'],
            agent=self.senior_story_writer_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


@CrewBase
class CreateContentCrew:
    """Create Content crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def senior_content_editor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['senior_content_editor'],
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

    @task
    def create_html(self) -> Task:
        return Task(
            config=self.tasks_config['create_html_task'],
            agent=self.senior_content_editor_agent(),
        )

    @task
    def qa_content(self) -> Task:
        return Task(
            config=self.tasks_config['qa_task'],
            agent=self.senior_content_editor_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


class StoryCreatorCrew():
    def __init__(self, request_data):
        self.request_data = request_data

    def run(self):
        # Analyze the request
        analysis = self.runAnalyzeRequestCrew(self.request_data)

        # Generate the story
        story_result = self.runGenerateStoryCrew(analysis)

        # Create HTML and QA
        html_content = self.runCreateContentCrew(story_result)

        return html_content

    def runAnalyzeRequestCrew(self, request_data):
        inputs = {
            "request_json": json.dumps(request_data)
        }
        analysis = AnalyzeRequestCrew().crew().kickoff(inputs=inputs)
        return str(analysis)

    def runGenerateStoryCrew(self, analysis):
        inputs = {
            "analysis": analysis
        }
        story = GenerateStoryCrew().crew().kickoff(inputs=inputs)
        return str(story)

    def runCreateContentCrew(self, story_result):
        inputs = {
            "story_result": story_result
        }
        result = CreateContentCrew().crew().kickoff(inputs=inputs)
        return str(result)