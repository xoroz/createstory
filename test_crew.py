#!/usr/bin/env python3
from crew import StoryCreatorCrew
import json

# Test data
test_data = {
    "request_id": "test_001",
    "parameters": {
        "age_range": "7-9",
        "theme": "space",
        "lesson": "friendship",
        "characters": "max the mouse, stella the star",
        "story_about": "max discovers a magical star",
        "length": "short",
        "language": "en",
        "ai_model": "google/gemini-2.5-flash"
    }
}

crew = StoryCreatorCrew(test_data)
result = crew.run()
print("Result:", result)