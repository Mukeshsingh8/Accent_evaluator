"""
English Accent Evaluator

A production-ready tool for analyzing English accents in video interviews.
Supports both rule-based and LLM-powered accent detection.
"""

__version__ = "1.0.0"
__author__ = "Accent Evaluator Team"

from .audio import extract_audio_from_video, extract_audio_features
from .transcription import transcribe_audio
from .accent import analyze_accent, generate_summary
from .llm import llm_accent_analysis
from .utils import setup_logging, get_logger

__all__ = [
    "extract_audio_from_video",
    "extract_audio_features", 
    "transcribe_audio",
    "analyze_accent",
    "generate_summary",
    "llm_accent_analysis",
    "setup_logging",
    "get_logger"
] 