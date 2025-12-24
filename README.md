## README.md

Project Structure

.
├── README.md
├── config.ini            # Application configuration (includes DB schema)
├── logs/                 # Log files
│   └── prompts/          # Detailed logs of AI requests
├── processed/            # Processed story requests
├── queue/                # Queue for story requests
├── requirements.txt      # Python dependencies
├── start.sh              # Start script for all services
      # Generated audio narration
├── story.json   # Story metadata (ratings, views)
├── story_processor.py    # Background story processing service