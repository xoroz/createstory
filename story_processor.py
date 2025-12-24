#!/usr/bin/env python3
import os
import re
import openai
import requests
import uuid
import threading
import logging
import json
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Create output directories if they don't exist
Path("stories").mkdir(exist_ok=True)
Path("stories/image").mkdir(exist_ok=True)
Path("stories/audio").mkdir(exist_ok=True)
Path("queue").mkdir(exist_ok=True)
Path("processed").mkdir(exist_ok=True)
Path("error").mkdir(exist_ok=True)

def generate_story_text(params, request_id):
    """Generate story text using OpenRouter for all providers.
    
    Args:
        params (dict): Parameters for story generation
        request_id (str): Request ID for logging
        
    Returns:
        tuple: (story_text, title, model, backend, username, email)
    """
    story_about = params.get('story_about', 'an adventure in an enchanted forest')
    language = params.get('language', 'en')
    ai_model = params.get('ai_model', 'gpt-3.5-turbo')  # Use ai_model for consistency
    backend = params.get('backend', 'openai')
    
    logger.info(f"Generating story about '{story_about}' in {language} using {backend}/{ai_model}")
    logger.info(f"Request parameters: backend={backend}, ai_model={ai_model}")
    
    # Get system prompt and user message from request data if available
    system_prompt = params.get('system_prompt', None)
    user_message = params.get('user_message', None)
    
    # Get username and email from params if available
    username = params.get('username', None)
    email = params.get('email', None)
    
    # ALWAYS use OpenRouter for story generation regardless of backend
    return generate_with_openrouter(story_about, language, ai_model, request_id, backend, system_prompt, user_message, username, email)

def generate_with_openai(story_about, language, model, request_id):
    """Generate story text using OpenAI API.
    
    Args:
        story_about (str): Brief description of the story
        language (str): Language code
        model (str): Model to use
        request_id (str): Request ID for logging
        
    Returns:
        tuple: (story_text, title, model, 'openai')
    """
    if not OPENAI_API_KEY:
        logger.error(f"OpenAI API key not found for request {request_id}")
        return None, None, model, 'openai'
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Build the prompt
    system_prompt = f"""You are StoryMagic, an expert children's story writer. 
Write a delightful children's story in {language} about {story_about}. 
The story should be engaging, age-appropriate, and between 500-800 words.
Include a title for the story.
Format your response as JSON with 'title' and 'story' fields.
The story should have 3-5 paragraphs with a clear beginning, middle, and end.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        response_data = json.loads(content)
        
        title = response_data.get('title', 'My Magical Story')
        story = response_data.get('story', '')
        
        logger.info(f"Successfully generated story with title: {title}")
        return story, title, model, 'openai'
        
    except Exception as e:
        logger.error(f"Error generating story with OpenAI: {str(e)}")
        return None, None, model, 'openai'

def generate_with_anthropic(story_about, language, model, request_id):
    """Generate story text using Anthropic API through OpenRouter.
    
    Args:
        story_about (str): Brief description of the story
        language (str): Language code
        model (str): Model to use
        request_id (str): Request ID for logging
        
    Returns:
        tuple: (story_text, title, model, 'anthropic')
    """
    logger.info(f"Using Anthropic (via OpenRouter) with model: {model}")
    
    # Ensure the model has the correct prefix
    if '/' not in model:
        model = f"anthropic/{model}"
    
    # Use OpenRouter for Anthropic models
    return generate_with_openrouter(story_about, language, model, request_id, 'anthropic')

def generate_with_mistral(story_about, language, model, request_id):
    """Generate story text using Mistral API through OpenRouter.
    
    Args:
        story_about (str): Brief description of the story
        language (str): Language code
        model (str): Model to use
        request_id (str): Request ID for logging
        
    Returns:
        tuple: (story_text, title, model, 'mistral')
    """
    logger.info(f"Using Mistral (via OpenRouter) with model: {model}")
    
    # Ensure the model has the correct prefix
    if '/' not in model:
        model = f"mistralai/{model}"
    
    # Use OpenRouter for Mistral models
    return generate_with_openrouter(story_about, language, model, request_id, 'mistral')

def generate_with_openrouter(story_about, language, model, request_id, original_backend='openrouter', system_prompt=None, user_message=None, username=None, email=None):
    """Generate story text using OpenRouter API for all models.
    
    Args:
        story_about (str): Brief description of the story
        language (str): Language code
        model (str): Model to use
        request_id (str): Request ID for logging
        original_backend (str): Original backend requested (for metadata)
        
    Returns:
        tuple: (story_text, title, model, original_backend)
    """
    if not OPENROUTER_API_KEY:
        logger.error(f"OpenRouter API key not found for request {request_id}")
        return None, None, model, original_backend
    
    # Ensure the model parameter includes the full provider path if needed
    if '/' not in model:
        # Check if this is a known model and prepend the provider
        if model.startswith('claude'):
            model = f"anthropic/{model}"
        elif model.startswith('mistral'):
            model = f"mistralai/{model}"
        elif model.startswith('gpt'):
            model = f"openai/{model}"
        # Add other model prefixes as needed
    
    logger.info(f"Using OpenRouter with model: {model}, language: {language}")
    
    # Use provided system prompt or create default
    if not system_prompt:
        # Build the prompt with stronger language enforcement
        system_prompt = f"""You are StoryMagic, an expert children's story writer. 
Write a delightful children's story ENTIRELY IN {language} about {story_about}. 
The story should be engaging, age-appropriate, and between 500-800 words.
Include a title for the story, also in {language}.
Format your response as JSON with 'title' and 'story' fields.
The story should have 3-5 paragraphs with a clear beginning, middle, and end.
IMPORTANT: Both the title and story MUST be written entirely in {language}.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Create messages array
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add user message if provided
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error from OpenRouter: {response.status_code} - {response.text}")
            return None, None, model, 'openrouter'
            
        response_data = response.json()
        
        # Check if response_data has the expected structure
        if 'choices' in response_data and isinstance(response_data['choices'], list) and len(response_data['choices']) > 0:
            if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                content = response_data['choices'][0]['message']['content']
            else:
                logger.error(f"Unexpected response structure from OpenRouter: {response_data}")
                return None, None, model, 'openrouter'
        else:
            logger.error(f"Unexpected response structure from OpenRouter: {response_data}")
            return None, None, model, 'openrouter'
        
        try:
            parsed_content = json.loads(content)
            title = parsed_content.get('title', 'My Magical Story')
            story = parsed_content.get('story', '')
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from OpenRouter response: {str(e)}")
            logger.error(f"Raw content: {content[:500]}")
            
            # Try to extract title and story using regex as fallback
            # First, try to find a title at the beginning of the text (common format)
            title_line_match = re.search(r'^Title:?\s*["\']?([^"\'\n]+)["\']?', content, re.IGNORECASE)
            if title_line_match:
                title = title_line_match.group(1).strip()
                logger.info(f"Extracted title from beginning of text: '{title}'")
            else:
                # Try JSON-style title extraction
                title_match = re.search(r'"title"\s*:\s*"([^"]+)"', content)
                title = title_match.group(1) if title_match else 'My Magical Story'
            
            # Try to extract story content
            story_match = re.search(r'"story"\s*:\s*"([^"]+)"', content)
            story = story_match.group(1) if story_match else ''
            
            if not story:
                # If regex fails too, use the raw content as the story
                # But try to remove the title line if it exists
                if title_line_match:
                    # Skip the title line and use the rest as the story
                    story_start = content.find('\n', title_line_match.start())
                    if story_start > 0:
                        story = content[story_start:].strip()
                    else:
                        story = content
                else:
                    story = content
                
                logger.warning("Using raw content as story text")
                
            # Log username and email to ensure they're preserved
            logger.info(f"Preserving username: {username} and email in fallback mechanism")
        
        # Get the actual model that responded
        actual_model = model
        if 'model' in response_data:
            actual_model = response_data['model']
        
        logger.info(f"Successfully generated story with title: {title} using model: {actual_model}")
        return story, title, actual_model, 'openrouter', username, email
        
    except Exception as e:
        logger.error(f"Error generating story with OpenRouter: {str(e)}")
        return None, None, model, 'openrouter', username, email

def generate_audio(text, language='en', request_id=None, enhanced_audio=False):
    """Generate audio for text using ElevenLabs or OpenAI.
    
    Args:
        text (str): Text to convert to audio
        language (str, optional): Language code. Defaults to 'en'.
        request_id (str, optional): Request ID for logging.
        enhanced_audio (bool, optional): Whether to use enhanced audio. Defaults to False.
        
    Returns:
        str: Path to audio file, or None if generation failed
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    audio_dir = Path("stories/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a filename using request_id
    audio_filename = f"{request_id}.mp3"
    audio_path = audio_dir / audio_filename
    
    if ELEVENLABS_API_KEY and enhanced_audio:
        # Use ElevenLabs for enhanced audio
        success = generate_audio_elevenlabs(text, audio_path, language, request_id)
    else:
        # Use OpenAI for standard audio
        success = generate_audio_openai(text, audio_path, language, request_id)
    
    if success:
        logger.info(f"Audio generated successfully: {audio_path}")
        return str(audio_filename)  # Return just the filename
    else:
        logger.error(f"Failed to generate audio for request {request_id}")
        return None

def generate_audio_openai(text, audio_path, language, request_id):
    """Generate audio using OpenAI TTS API.
    
    Args:
        text (str): Text to convert to audio
        audio_path (Path): Path to save audio file
        language (str): Language code
        request_id (str): Request ID for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not OPENAI_API_KEY:
        logger.error(f"OpenAI API key not found for request {request_id}")
        return False
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Choose voice based on language
    voice = "alloy"  # default voice
    if language.startswith("es"):
        voice = "shimmer"
    elif language.startswith("fr"):
        voice = "echo"
    elif language.startswith("de"):
        voice = "onyx"
    elif language.startswith("it"):
        voice = "nova"
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        with open(audio_path, "wb") as f:
            response.stream_to_file(audio_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating audio with OpenAI: {str(e)}")
        return False

def generate_audio_elevenlabs(text, audio_path, language, request_id):
    """Generate audio using ElevenLabs API.
    
    Args:
        text (str): Text to convert to audio
        audio_path (Path): Path to save audio file
        language (str): Language code
        request_id (str): Request ID for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not ELEVENLABS_API_KEY:
        logger.error(f"ElevenLabs API key not found for request {request_id}")
        return False
    
    try:
        from elevenlabs import generate, save
        from elevenlabs import set_api_key
        
        set_api_key(ELEVENLABS_API_KEY)
        
        # Choose voice based on language
        voice = "Bella"  # default voice for English
        if language.startswith("es"):
            voice = "Antonio"
        elif language.startswith("fr"):
            voice = "Josephine"
        elif language.startswith("de"):
            voice = "Wolfgang"
        elif language.startswith("it"):
            voice = "Matilda"
        
        # Generate audio
        audio = generate(
            text=text,
            voice=voice,
            model="eleven_multilingual_v2"
        )
        
        # Save the audio file
        save(audio, audio_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating audio with ElevenLabs: {str(e)}")
        return False

def generate_images(story_id, story_brief, story_content, request_id=None):
    """Generate 3 images for a story and save to the stories/image folder.
    
    Args:
        story_id (str): ID of the story (for naming the image files)
        story_brief (str): Brief description of the story
        story_content (str): Full content of the story
        request_id (str, optional): Request ID for logging purposes
    
    Returns:
        list: List of relative paths to the generated images, or None if generation failed
    """
    logger.info(f"Generating images for story {story_id}")
    
    # Get API key from environment variables
    if not OPENAI_API_KEY:
        logger.error(f"OPENAI_API_KEY not found in environment variables for request {request_id}")
        return None

    # Set up OpenAI API client
    openai.api_key = OPENAI_API_KEY
    
    # Define constants
    IMAGE_OUTPUT_DIR = Path("stories/image")
    IMAGE_SIZE = "1024x1024"  # Must use OpenAI's supported sizes
    IMAGE_QUALITY = "standard"  # Options: standard, hd
    
    # Create output directory if it doesn't exist
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate custom image prompts using GPT
    prompts = generate_image_prompts(story_brief, story_content)
    if not prompts:
        logger.error(f"Failed to generate image prompts for story {story_id}")
        return None
        
    # Define image filenames
    image_filenames = [
        f"{story_id}-1.png",
        f"{story_id}-2.png",
        f"{story_id}-3.png"
    ]
    
    # Generate and save each image
    success_count = 0
    for i, prompt in enumerate(prompts, 1):
        if generate_and_save_image(prompt, story_id, i):
            success_count += 1
    
    # Return image filenames if at least one image was generated successfully
    if success_count > 0:
        logger.info(f"Successfully generated {success_count} images for story {story_id}")
        return image_filenames[:success_count]  # Return only the successful ones
    else:
        logger.error(f"Failed to generate any images for story {story_id}")
        return None

def generate_image_prompts(story_brief, story_content):
    """Use GPT-3.5-turbo to generate custom image prompts based on the story."""
    logger.info("Generating custom image prompts using GPT-3.5-turbo")
    
    # Extract a preview of the story content (beginning, middle, and end)
    content_length = len(story_content)
    beginning = story_content[:min(500, content_length)]
    middle_start = max(0, content_length // 3 - 250)
    middle = story_content[middle_start:min(middle_start + 500, content_length)]
    end_start = max(0, content_length - 500)
    ending = story_content[end_start:]
    
    # Create a system prompt that explains exactly what we want
    system_prompt = """
    You are a children's book illustrator who creates beautiful, 
    text-free illustrations for children's stories. Your task is to create 
    3 detailed image prompts for a children's story - one for the beginning, 
    one for the middle, and one for the end of the story.

    IMPORTANT GUIDELINES:
    1. Each prompt should describe a purely visual scene with NO TEXT elements.
    2. Focus on characters, emotions, settings, and actions.
    3. Avoid mentioning any text, labels, words, or writing in your prompts.
    4. Make the prompts suitable for children's book illustrations.
    5. Each prompt should be 3-5 sentences and focus on visual storytelling.
    6. Embrace colorful, warm, friendly illustration style.
    """
    
    # Create the user prompt with story information
    user_prompt = f"""
    Story Brief: {story_brief}
    
    Beginning of Story: {beginning}
    
    Middle of Story: {middle}
    
    End of Story: {ending}
    
    Based on this story, create 3 purely visual illustration prompts:
    1. For the beginning of the story (introducing characters and setting)
    2. For the middle of the story (showing the main adventure or conflict)
    3. For the end of the story (showing the resolution)
    
    Format your response as a JSON object with keys "prompt1", "prompt2", and "prompt3".
    Each prompt value should be a detailed description for a text-free illustration.
    Make sure each prompt has NO instructions to create text or words in the image.
    """
    
    try:
        # Call GPT-3.5-turbo to generate the prompts
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Extract the generated prompts
        prompt_text = response.choices[0].message.content
        
        try:
            # Parse the JSON response
            prompts = json.loads(prompt_text)
            
            # Ensure we have all three prompts
            if "prompt1" in prompts and "prompt2" in prompts and "prompt3" in prompts:
                logger.info("Successfully generated custom image prompts")
                return [prompts["prompt1"], prompts["prompt2"], prompts["prompt3"]]
            else:
                logger.warning("Generated prompts are missing some keys. Using partial results.")
                # Return whatever prompts we have, with defaults for missing ones
                default_prompt = "A colorful illustration for a children's story without any text or words."
                return [
                    prompts.get("prompt1", default_prompt),
                    prompts.get("prompt2", default_prompt),
                    prompts.get("prompt3", default_prompt)
                ]
                
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON response from GPT. Using raw text.")
            # Extract prompts using regex if JSON parsing fails
            prompt_matches = re.findall(r'"prompt\d+"\s*:\s*"([^"]+)"', prompt_text)
            if len(prompt_matches) >= 3:
                return prompt_matches[:3]
            else:
                # Fall back to default prompts
                logger.warning("Failed to extract prompts from GPT response. Using default prompts.")
                return generate_default_prompts(story_brief, story_content)
                
    except Exception as e:
        logger.error(f"Failed to generate prompts with GPT: {str(e)}")
        # Fall back to default prompts
        return generate_default_prompts(story_brief, story_content)

def generate_default_prompts(story_brief, story_content):
    """Generate default prompts if GPT prompt generation fails."""
    logger.info("Using default prompt generation logic based on story content")
    
    # Default prompts that will adapt to the story brief
    prompt1 = (
        f"A colorful, kid-friendly illustration showing the beginning of a story about '{story_brief}'. "
        "The scene introduces the main characters in their initial setting, showing their emotions and surroundings. "
        "The illustration has a warm, inviting style with no text elements."
    )
    
    prompt2 = (
        f"A dynamic illustration showing the middle of a story about '{story_brief}'. "
        "The main characters are shown in the midst of their adventure or challenge. "
        "The scene captures action and emotion in a colorful, child-friendly style without any text."
    )
    
    prompt3 = (
        f"A heartwarming illustration showing the conclusion of a story about '{story_brief}'. "
        "The scene shows the resolution with characters expressing joy, relief, or satisfaction. "
        "The illustration has a peaceful, satisfying quality in a colorful style with no text elements."
    )
    
    return [prompt1, prompt2, prompt3]

def generate_and_save_image(prompt, story_id, image_number, max_retries=3):
    """Generate image using OpenAI DALL-E API and save it to disk."""
    # Create directory if it doesn't exist
    output_dir = Path("stories/image")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use PNG format to preserve quality
    image_path = output_dir / f"{story_id}-{image_number}.png"
    
    # Create a very strong anti-text prompt with specific instruction pattern
    # that addresses how DALL-E tends to add text
    anti_text_prefix = (
        "Create a children's book illustration with absolutely NO TEXT, NO WORDS, NO LETTERS, "
        "and NO WRITING of any kind. Do not include any alphabetic characters, numbers, "
        "symbols, signs, banners, scrolls, books with visible text, labels, or anything "
        "that resembles writing. Create ONLY visual imagery - no logos, no signatures, no watermarks. "
        "This is for a wordless picture book where the story is told ENTIRELY through images. "
    )
    
    # Combine the anti-text prefix with the specific prompt
    modified_prompt = anti_text_prefix + prompt
    
    logger.info(f"Generating image {image_number}")
    logger.debug(f"Using prompt: {modified_prompt[:150]}...")
    IMAGE_OUTPUT_DIR = Path("stories/image")
    IMAGE_SIZE = "1024x1024"  # Must use OpenAI's supported sizes
    IMAGE_QUALITY = "standard"  # Options: standard, hd
    for attempt in range(max_retries):
        try:
            # Start timer for performance tracking
            start_time = time.time()
            
            # Generate image using DALL-E
            response = openai.images.generate(
                model="dall-e-3",
                prompt=modified_prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
                style="vivid",  # More colorful children's book style
            )
            
            generation_time = time.time() - start_time
            logger.debug(f"Image generation took {generation_time:.2f} seconds")
            
            # Get the image URL
            image_url = response.data[0].url
            logger.debug(f"Image URL received")
            
            # Download the image
            download_start = time.time()
            image_response = requests.get(image_url, timeout=30)
            download_time = time.time() - download_start
            logger.debug(f"Image download took {download_time:.2f} seconds")
            
            if image_response.status_code != 200:
                logger.error(f"Failed to download image from URL. Status code: {image_response.status_code}")
                continue
            
            # Save the original image without compression
            with open(image_path, "wb") as f:
                f.write(image_response.content)
            
            # Verify that the image opens correctly
            try:
                saved_image = Image.open(image_path)
                width, height = saved_image.size
                file_size = os.path.getsize(image_path) / 1024  # Size in KB
                logger.debug(f"Image verified - dimensions: {width}x{height}, file size: {file_size:.2f}KB")
                saved_image.close()  # Properly close the image
            except Exception as e:
                logger.error(f"Saved image cannot be opened: {str(e)}")
                
                # Save the raw content as a backup
                backup_path = output_dir / f"{story_id}-{image_number}.backup.png"
                with open(backup_path, "wb") as f:
                    f.write(image_response.content)
                logger.debug(f"Saved backup image to {backup_path}")
                
                # Try to open the image directly from memory to diagnose the issue
                try:
                    memory_image = Image.open(BytesIO(image_response.content))
                    logger.debug(f"Image opens correctly from memory but not from disk. File system issue?")
                    memory_image.close()
                except Exception as e2:
                    logger.debug(f"Image also fails to open from memory: {str(e2)}")
                    continue  # Try again
            
            logger.info(f"Image {image_number} saved locally to {image_path}")
            
            # Clear any cached references to the remote URL
            del image_response
            return True
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            if attempt < max_retries - 1:
                logger.debug(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)  # Small delay before retry
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return False
                
        except Exception as e:
            logger.error(f"Exception while generating image {image_number}: {str(e)}")
            if attempt < max_retries - 1:
                logger.debug(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return False
    
    return False

def story_to_html(story, title, request_id, audio_path, image_paths=None, language='en', 
                  backend='openai', model='gpt-3.5-turbo', story_about=None, username=None):
    """
    Convert a story to an HTML document, optionally including images.
    
    Args:
        story (str): The story text
        title (str): Title of the story
        request_id (str): Request ID for the story
        audio_path (str): Path to the audio file
        image_paths (list, optional): List of paths to image files to include
        language (str, optional): Language code of the story. Defaults to 'en'.
        backend (str, optional): AI provider used. Defaults to 'openai'.
        model (str, optional): Model used. Defaults to 'gpt-3.5-turbo'.
        story_about (str, optional): Brief description of the story
        username (str, optional): Username of the story creator
        
    Returns:
        str: HTML content for the story
    """
    # Log the username parameter
    logger.info(f"story_to_html called with username: '{username}'")
    # Validate story content
    if not story or len(story) < 50:  # Arbitrary minimum length
        logger.error(f"Story content is too short or empty: '{story}'")
        # Use a fallback message if story is empty
        if not story:
            story = f"[Error: Story generation failed. Please try again.]"
    
    # Improved paragraph splitting - handle different paragraph separators
    # First, normalize line endings
    normalized_story = story.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by double newlines (standard paragraph separator)
    paragraphs = normalized_story.split('\n\n')
    
    # If we only got one paragraph, try splitting by single newlines
    if len(paragraphs) <= 1 and '\n' in normalized_story:
        paragraphs = normalized_story.split('\n')
    
    # Filter out empty paragraphs and trim whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Debug logging
    logger.info(f"Story split into {len(paragraphs)} paragraphs")
    if len(paragraphs) <= 1:
        logger.warning(f"Story may not be properly formatted. First 200 chars: {story[:200]}")
    
    num_paragraphs = len(paragraphs)
    html_content_parts = []

    # Interleave images with paragraphs
    img_idx = 0
    img_placement_indices = []
    if image_paths and num_paragraphs > 1:
        # Place after paragraph 1 (index 0), middle (index floor(n/2)-1), and before last (index n-2)
        img_placement_indices.append(0)  # After first paragraph
        if num_paragraphs > 2:
            img_placement_indices.append(max(1, num_paragraphs // 2 - 1))  # After middle-ish paragraph
        if num_paragraphs > 3:
            img_placement_indices.append(max(1, num_paragraphs - 2))  # Before last paragraph (after n-2)
        # Remove duplicates and sort
        img_placement_indices = sorted(list(set(img_placement_indices)))

    # Format the AI provider name to look nicer
    provider_display = {
        'openai': 'OpenAI',
        'deepseek': 'DeepSeek',
        'anthropic': 'Claude',
        'mistral': 'Mistral AI'
    }.get(backend, backend.title())
    
    # Current date
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Add story_about section (if provided)
    story_about_html = ""
    if story_about:
        story_about_html = f"""
        <div class="story-brief">
            <h3>Story Brief</h3>
            <p class="story-about">{story_about}</p>
        </div>
        """
    
    # Add audio player if audio is available
    audio_html = ""
    if audio_path:
        audio_html = f"""
        <div class="audio-player">
            <h3>Listen to the story:</h3>
            <audio controls style="width: 100%; margin: 20px 0;">
                <source src="/audio/{audio_path}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
    
    # Add username info if provided
    username_html = ""
    if username:
        username_html = f"""
        <div class="story-author">
            <span class="author-label">Created by:</span>
            <span class="author-name">{username}</span>
        </div>
        """

    # Build the content with paragraphs and images
    for i, p in enumerate(paragraphs):
        html_content_parts.append(f"<p>{p}</p>")
        if image_paths and i in img_placement_indices and img_idx < len(image_paths):
            # Use the direct-image path for better performance with Caddy
            img_filename = image_paths[img_idx]
            img_html = f'<div class="story-image-container"><img src="/direct-image/{img_filename}" alt="Story illustration {img_idx+1}" class="story-image"></div>'
            html_content_parts.append(img_html)
            img_idx += 1

    # Ensure all images are placed if paragraphs were too few
    while image_paths and img_idx < len(image_paths):
        img_filename = image_paths[img_idx]
        # Add remaining images at the end with correct path
        img_html = f'<div class="story-image-container"><img src="/direct-image/{img_filename}" alt="Story illustration {img_idx+1}" class="story-image"></div>'
        html_content_parts.append(img_html)
        img_idx += 1
    
    # Combine all parts of the content
    story_content = "\n".join(html_content_parts)
    
    # Build the complete HTML
    html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/css/story.css">
</head>
<body>
    <div class="story-container">
        <header>
            <h1 class="story-title">{title}</h1>
            <div class="story-meta">
                <span class="story-date">{current_date}</span>
                <span class="story-provider">Created by {provider_display} using {model}</span>
            </div>
            {username_html}
        </header>
        
        <div class="story-content-wrapper">
            {audio_html}
            {story_about_html}
            <div class="story-content">
{story_content}
            </div>
        </div>
        
        <div class="actions">
            <a href="/create" class="button">Create Another Story</a>
            <a href="/stories" class="button">View All Stories</a>
        </div>
    </div>
    
    <footer>
        <p>StoryMagic - AI-Powered Stories for Little Imaginations</p>
        <p>&copy; {datetime.now().year} StoryMagic</p>
    </footer>
</body>
</html>"""

    return html

def generate_story(request_data):
    """Main function to generate a complete story with audio and images."""
    # Extract parameters
    params = request_data.get('parameters', {})
    story_about = params.get('story_about', '')
    language = params.get('language', 'en')
    request_id = request_data.get('id', str(uuid.uuid4()))
    username = params.get('username', '')
    
    # Make sure backend is passed to params
    backend = request_data.get('backend', 'openai')
    params['backend'] = backend
    
    logger.info(f"Processing story request {request_id} about '{story_about}' with backend {backend}")
    
    # Get system prompt and user message from request data
    system_prompt = request_data.get('prompts', {}).get('system_prompt', '')
    user_message = request_data.get('prompts', {}).get('user_message', '')
    
    # Add these to params for use in generate_story_text
    if system_prompt:
        params['system_prompt'] = system_prompt
    if user_message:
        params['user_message'] = user_message
    
    # Log the prompts
    logger.info(f"Using system prompt: {system_prompt[:100]}...")
    logger.info(f"Using user message: {user_message[:100]}...")
    
    # Get the title from parameters or use a default
    user_title = params.get('title')
    
    # Extract username and email from request data
    username = request_data.get('username', '')
    email = request_data.get('email', '')
    
    # Add username and email to params
    params['username'] = username
    params['email'] = email
    
    # Generate the story text
    story_text, ai_title, model, backend, username, email = generate_story_text(params, request_id)
    if not story_text:
        logger.error(f"Failed to generate story text for request {request_id}")
        return {"status": "error", "message": "Failed to generate story"}
    
    # Use the user-provided title if available, otherwise use the AI-generated title
    title = user_title if user_title else ai_title
    
    # Log which title is being used
    if user_title:
        logger.info(f"Using user-provided title: '{title}'")
    else:
        logger.info(f"Using AI-generated title: '{title}'")
    
    # Run audio and image generation in parallel
    audio_path = None
    image_paths = None
    
    # Define worker functions
    def audio_worker():
        nonlocal audio_path
        enhanced_audio = params.get('enhanced_audio', False)
        audio_path = generate_audio(story_text, language, request_id, enhanced_audio=enhanced_audio)
    
    def image_worker():
        nonlocal image_paths
        image_paths = generate_images(
            story_id=request_id,
            story_brief=story_about,
            story_content=story_text,
            request_id=request_id
        )
    
    # Start threads for parallel processing
    threads = []
    
    # Only start audio thread if audio is enabled
    if params.get('enable_audio', True):
        logger.info(f"Starting audio generation for request {request_id}")
        audio_thread = threading.Thread(target=audio_worker)
        audio_thread.start()
        threads.append(audio_thread)
    else:
        logger.info(f"Audio generation disabled for request {request_id}")
    
    # Only start image thread if images are enabled
    if params.get('enable_images', True):
        logger.info(f"Starting image generation for request {request_id}")
        image_thread = threading.Thread(target=image_worker)
        image_thread.start()
        threads.append(image_thread)
    else:
        logger.info(f"Image generation disabled for request {request_id}")
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Create HTML
    logger.info(f"Generating HTML for request {request_id}")
    html_content = story_to_html(
        story=story_text,
        title=title,
        request_id=request_id,
        audio_path=audio_path,
        image_paths=image_paths,
        language=language,
        backend=backend,
        model=model,
        story_about=story_about,
        username=username
    )
    
    # Save HTML file
    filename = f"{title.lower().replace(' ', '-')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    filepath = os.path.join("stories", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Story generation complete for request {request_id}, saved as {filepath}")
    
    return {
        "status": "success",
        "story_path": filepath,
        "audio_path": audio_path,
        "image_paths": image_paths,
        "title": title
    }

def process_request(request_path):
    """Process a request from the queue."""
    request_id = os.path.basename(request_path).split('.')[0]
    logger.info(f"Processing request: {request_id}")
    
    try:
        # Read the request file
        with open(request_path, 'r') as f:
            request_data = json.load(f)
        
        # Debug log the request data
        logger.info(f"Request data: backend={request_data.get('backend')}, model={request_data.get('parameters', {}).get('ai_model')}")
        
        # Generate the story
        result = generate_story(request_data)
        
        # Add result data to request data
        request_data['result'] = result
        request_data['status'] = 'completed' if result.get('status') == 'success' else 'error'
        request_data['completed_at'] = datetime.now().isoformat()
        
        # Add output file and audio file to the request data for easier access
        if result.get('status') == 'success':
            request_data['output_file'] = os.path.basename(result.get('story_path', ''))
            request_data['audio_file'] = result.get('audio_path')
            request_data['image_paths'] = result.get('image_paths')
            
            # Ensure username and email are preserved in the processed JSON
            if 'username' not in request_data and request_data.get('parameters', {}).get('username'):
                request_data['username'] = request_data['parameters']['username']
            if 'email' not in request_data and request_data.get('parameters', {}).get('email'):
                request_data['email'] = request_data['parameters']['email']
        
        # Save processed request
        processed_path = os.path.join('processed', f"{request_id}.json")
        with open(processed_path, 'w') as f:
            json.dump(request_data, f, indent=2)
        
        # Populate the database with the story data
        if result.get('status') == 'success':
            try:
                # Import here to avoid circular imports
                from db_utils import populate_story_db
                
                # Populate the database
                db_result = populate_story_db(processed_path)
                if db_result:
                    logger.info(f"Successfully added story {request_id} to database")
                else:
                    logger.error(f"Failed to add story {request_id} to database")
            except Exception as db_error:
                logger.error(f"Error adding story to database: {str(db_error)}")
        
        # Remove the request file from queue
        try:
            os.remove(request_path)
            logger.info(f"Request {request_id} processed successfully and removed from queue")
        except Exception as e:
            logger.error(f"Error removing request from queue: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        
        # Move request to error folder
        try:
            error_path = os.path.join('error', f"{request_id}.json")
            if not os.path.exists(error_path):
                os.rename(request_path, error_path)
                logger.info(f"Request {request_id} moved to error folder")
                
                # Add error info
                with open(error_path, 'r') as f:
                    error_data = json.load(f)
                
                error_data['status'] = 'error'
                error_data['error_message'] = str(e)
                error_data['error_time'] = datetime.now().isoformat()
                
                with open(error_path, 'w') as f:
                    json.dump(error_data, f, indent=2)
            else:
                logger.warning(f"Error file already exists for request {request_id}")
                os.remove(request_path)
                
        except Exception as move_error:
            logger.error(f"Error moving request to error folder: {str(move_error)}")
        
        return False

def main():
    """Main function to monitor the queue folder for requests."""
    logger.info("Story generator service started. Monitoring queue folder.")
    
    while True:
        try:
            # Check for request files in queue folder
            requests = [f for f in os.listdir('queue') if f.endswith('.json')]
            
            if requests:
                # Process the oldest request
                requests.sort(key=lambda x: os.path.getmtime(os.path.join('queue', x)))
                next_request = os.path.join('queue', requests[0])
                process_request(next_request)
            else:
                # No requests, sleep for a bit
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # Sleep longer on error

if __name__ == "__main__":
    main()
