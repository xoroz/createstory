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
from crew import StoryCreatorCrew

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create output directories
Path("logs").mkdir(exist_ok=True)
Path("story").mkdir(exist_ok=True)
Path("story/image").mkdir(exist_ok=True)
Path("story/audio").mkdir(parents=True, exist_ok=True)
Path("queue").mkdir(exist_ok=True)
Path("processed").mkdir(exist_ok=True)
Path("error").mkdir(exist_ok=True)




def log_success(request_id, title, model, params, html_content):
    """Create story-<id>.html file in processed folder"""
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

        # Generate story using CrewAI crew
        crew = StoryCreatorCrew(request_data)
        html_content = crew.run()

        if not html_content:
            print(f"Failed to generate story for request {request_id}")
            return False

        # Extract title and story from HTML for logging
        import re
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else 'My Magical Story'

        story_match = re.search(r'<div class="story-content">(.*?)</div>', html_content, re.DOTALL)
        story_text = story_match.group(1).replace('<br>', '\n') if story_match else ''

        model = params.get('ai_model', 'google/gemini-2.5-flash')
        backend = 'openrouter'

        # Save HTML file as story-<id>.html
        filename = f"story-{request_id}.html"
        filepath = os.path.join("story", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Story generated successfully! Saved as: {filepath}")

        # Log success to process.log
        log_success(request_id, title, model, params, html_content)

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