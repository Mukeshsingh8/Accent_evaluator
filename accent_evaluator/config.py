"""Configuration and constants for accent evaluator."""
import os
from typing import Dict, List

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Feature Flags
ENABLE_LLM_ANALYSIS = os.getenv("ENABLE_LLM_ANALYSIS", "true").lower() == "true"
ENABLE_RULE_BASED_ANALYSIS = os.getenv("ENABLE_RULE_BASED_ANALYSIS", "true").lower() == "true"

# Audio Processing
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "300"))  # 5 minutes
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Rate Limiting
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "100"))

# Error Messages
ERROR_MESSAGES = {
    "invalid_url": "Please provide a valid video URL (YouTube, Loom, or direct MP4).",
    "audio_extraction_failed": "Failed to extract audio from the video. Please check the URL and try again.",
    "transcription_failed": "Failed to transcribe the audio. Please ensure the video contains clear English speech.",
    "feature_extraction_failed": "Failed to analyze audio features. Please try again.",
    "llm_analysis_failed": "LLM analysis failed. Please check your API key and try again.",
    "rate_limit_exceeded": "Rate limit exceeded. Please try again later.",
    "invalid_api_key": "Invalid OpenAI API key. Please check your credentials.",
    "file_too_large": f"Video file is too large. Maximum duration is {MAX_AUDIO_DURATION} seconds.",
    "unsupported_format": "Unsupported video format. Please use MP4, AVI, MOV, or MKV.",
}

# Accent Keywords
ACCENT_KEYWORDS = {
    "American": [
        "r", "t", "d", "water", "better", "matter", "letter", "butter",
        "flap", "rhotic", "cot-caught", "father-bother", "merry-marry"
    ],
    "British": [
        "non-rhotic", "received pronunciation", "rp", "queen's english",
        "posh", "cockney", "estuary", "glottal stop", "t-glottalisation"
    ],
    "Australian": [
        "strine", "broad australian", "general australian", "cultivated australian",
        "rising intonation", "question intonation", "mate", "g'day"
    ],
    "Canadian": [
        "canadian raising", "about", "house", "out", "north", "force",
        "cot-caught merger", "canadian shift"
    ],
    "Indian": [
        "indian english", "hindi influence", "retroflex", "aspirated",
        "tamil influence", "telugu influence", "bengali influence"
    ]
}

# LLM Prompt Templates
LLM_PROMPT_TEMPLATE = """
You are an expert linguist and speech analyst specializing in English accent detection. 
Given the following transcript and audio features, identify the speaker's English accent.

Available accents: American, British, Australian, Canadian, Indian, Other

Transcript:
{transcript}

Audio features:
{audio_features}

Please respond in the following JSON format:
{{
  "accent": "accent_name",
  "confidence": confidence_score_0_to_100,
  "explanation": "detailed_explanation_of_why_this_accent_was_detected"
}}

Focus on:
- Pronunciation patterns
- Vocabulary choices
- Intonation patterns
- Regional speech markers
- Audio feature correlations
"""

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/accent_evaluator.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "accent_evaluator": {
            "level": "DEBUG" if DEBUG else "INFO",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
} 