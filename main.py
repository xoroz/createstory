#!/usr/bin/env python3
"""
Story Creator - CrewAI-powered story generation
Phase 1 Complete: Text generation with CrewAI agents
"""
import os
import json
import time
import re
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# CrewAI imports for AI orchestration
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create output directories
Path("logs").mkdir(exist_ok=True)
Path("stories").mkdir(exist_ok=True)
Path("stories/image").mkdir(exist_ok=True)
Path("stories/audio").mkdir(parents=True, exist_ok=True)
Path("queue").mkdir(exist_ok=True)
Path("processed").mkdir(exist_ok=True)
Path("error").mkdir(exist_ok=True)


class StoryWriter:
    """CrewAI agent specialized for children's stories"""

    def __init__(self):
        self.agent = Agent(
            role="Master Storywriter for Children",
            goal="Create engaging, age-appropriate children's stories that captivate young imaginations",
            backstory="""You are a master's level storyteller with 15+ years specializing in educational
            and entertaining children's literature. You excel at creating original stories tailored to
            specific age groups, incorporating requested themes while ensuring content is safe,
            educational, and emotionally engaging. You use clear, vivid language appropriate for each
            child's age level and always focus on positive values like friendship, family, and perseverance."""
        )


def generate_story_text(params, request_id):
    """
    Generate story text using CrewAI crew orchestration through OpenRouter

    Args:
        params (dict): Story parameters including theme, characters, language, etc.
        request_id (str): Unique request identifier for logging

    Returns:
        tuple: (story_text, title, model, 'openrouter')
    """
    # Extract story parameters
    story_about = params.get('story_about', 'an adventure in a magical world')
    language = params.get('language', 'en')
    ai_model = params.get('ai_model', 'openai/gpt-3.5-turbo')
    age_range = params.get('age_range', '7-9')

    # Create CrewAI story writer agent
    story_agent = StoryWriter()

    # Create LangChain LLM for OpenRouter integration
    llm = ChatOpenAI(
        model=ai_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    # Configure the agent with the LLM
    story_agent.agent.llm = llm

    # Create task for story generation
    generate_story_task = Task(
        description=f"""
        Create an engaging children's story for age range {age_range}.

        Story requirements:
        - Theme: {story_about}
        - Language: {language}
        - Age-appropriate content with educational value
        - Clear beginning, middle, and end structure
        - Engaging characters and plot development
        - Word count: 500-800 words
        - Include a title in {language}

        Format the response as JSON with keys: "title" and "story"
        """,
        agent=story_agent.agent,
        expected_output="Complete children's story in JSON format with title and story content"
    )

    # Create crew to manage the story generation
    crew = Crew(
        agents=[story_agent.agent],
        tasks=[generate_story_task],
        verbose=True
    )

    try:
        # Execute crew and get result
        result = crew.kickoff()

        if result and result.raw:
            # Parse the JSON response
            try:
                parsed = json.loads(result.raw.strip())
                title = parsed.get('title', 'My Magical Story')
                story_text = parsed.get('story', '')

                return story_text, title, ai_model, 'openrouter'
            except json.JSONDecodeError:
                # Fallback: try to extract title and story manually
                content = result.raw
                title_match = re.search(r'"title"\s*:\s*"([^"]+)"', content)
                story_match = re.search(r'"story"\s*:\s*"([^"]+)"', content)

                title = title_match.group(1).strip() if title_match else 'My Magical Story'
                story_text = story_match.group(1).strip() if story_match else content

                return story_text, title, ai_model, 'openrouter'
        else:
            return None, None, ai_model, 'openrouter'

    except Exception as e:
        print(f"Error in CrewAI story generation: {e}")
        return None, None, ai_model, 'openrouter'


def story_to_html(story, title, request_id, audio_path=None, image_paths=None, language='en', **kwargs):
    """
    Convert story to HTML document
    """
    # Basic HTML template
    html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .story-title {{ color: #2e7d32; text-align: center; }}
        .story-content {{ max-width: 800px; margin: 0 auto; }}
        .story-meta {{ text-align: center; color: #666; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1 class="story-title">{title}</h1>
    <div class="story-meta">
        <p>Created with CrewAI • {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    <div class="story-content">
        {story.replace(chr(10), '<br>')}
    </div>
    <div class="actions" style="text-align: center; margin-top: 40px;">
        <a href="/create" style="background: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Create Another Story</a>
    </div>
</body>
</html>"""
    return html


def log_success(request_id, title, filepath, model, params):
    """Create story-<id>.html file in processed folder"""
    # Create HTML content
    html_content = story_to_html(story_text, title, request_id, language=params.get('language', 'en'))

    # Save to processed folder with story-<id>.html naming
    filename = f"story-{request_id}.html"
    filepath = os.path.join("processed", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)


def process_story_request(request_file_path):
    """Process a single story request from JSON file"""
    try:
        with open(request_file_path, 'r') as f:
            request_data = json.load(f)

        params = request_data.get('parameters', {})
        request_id = params.get('request_id', str(uuid.uuid4()))

        print(f"Processing story request {request_id}")

        # Generate story text using CrewAI
        story_text, title, model, backend = generate_story_text(params, request_id)

        if not story_text:
            print(f"Failed to generate story for request {request_id}")
            return False

        # Create HTML output
        html_content = story_to_html(story_text, title, request_id, language=params.get('language', 'en'))

        # Save HTML file as story-<id>.html
        filename = f"story-{request_id}.html"
        filepath = os.path.join("stories", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Story generated successfully! Saved as: {filepath}")

        # Log success to process.log
        log_success(request_id, title, filepath, model, params)

        # Move request to processed folder
        processed_path = f"processed/{os.path.basename(request_file_path)}"
        os.rename(request_file_path, processed_path)

        return True

    except Exception as e:
        print(f"Error processing request: {e}")
        # Log error to process.log
        with open("logs/process.log", "a") as log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_file.write(f"[{timestamp}] ERROR - Request {request_id}: {str(e)}\n")
            log_file.write(f"- Original file: {request_file_path}\n\n")

        # Move to error folder if available
        error_path = f"error/{os.path.basename(request_file_path)}"
        if os.path.exists(request_file_path):
            os.rename(request_file_path, error_path)
        return False


def main():
    """Monitor queue folder and process story requests"""
    print("Story Creator (CrewAI) - Monitoring queue folder...")
    print("Drop JSON files in 'queue/' folder to generate stories!")

    while True:
        try:
            # Check for JSON request files
            queue_files = list(Path("queue").glob("*.json"))

            if queue_files:
                for json_file in queue_files:
                    print(f"Found request file: {json_file.name}")
                    process_story_request(str(json_file))

            time.sleep(1)  # Check every second

        except KeyboardInterrupt:
            print("\nShutting down story creator...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()