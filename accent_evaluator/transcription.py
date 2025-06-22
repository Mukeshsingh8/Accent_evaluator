"""Transcription logic using Whisper."""
from typing import Optional
import whisper
import time

from .utils import get_logger, generate_request_id
from .config import WHISPER_MODEL, ERROR_MESSAGES

logger = get_logger("transcription")

_model: Optional[whisper.Whisper] = None

def get_model() -> whisper.Whisper:
    """Get or load Whisper model with singleton pattern."""
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        start_time = time.time()
        _model = whisper.load_model(WHISPER_MODEL)
        logger.info(f"Whisper model loaded in {time.time() - start_time:.2f}s")
    return _model

def transcribe_audio(audio_file: str, request_id: str = None) -> str:
    """
    Transcribe audio using Whisper. Returns the transcription text.
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Starting audio transcription")
    
    try:
        start_time = time.time()
        model = get_model()
        
        logger.debug(f"[{request_id}] Transcribing audio file: {audio_file}")
        result = model.transcribe(audio_file)
        transcription = result["text"].strip()
        
        # Validate transcription
        if not transcription:
            logger.warning(f"[{request_id}] Empty transcription result")
            raise Exception("No speech detected in the audio")
        
        logger.info(f"[{request_id}] Transcription completed in {time.time() - start_time:.2f}s")
        logger.debug(f"[{request_id}] Transcription length: {len(transcription)} characters")
        
        return transcription
        
    except Exception as e:
        logger.error(f"[{request_id}] Transcription failed: {str(e)}")
        raise Exception(f"Error transcribing audio: {str(e)}")

def get_transcription_language(audio_file: str, request_id: str = None) -> str:
    """
    Detect the language of the audio using Whisper.
    Returns language code (e.g., 'en', 'es', 'fr').
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Detecting language")
    
    try:
        model = get_model()
        result = model.transcribe(audio_file, task="transcribe")
        language = result.get("language", "unknown")
        
        logger.info(f"[{request_id}] Detected language: {language}")
        return language
        
    except Exception as e:
        logger.error(f"[{request_id}] Language detection failed: {str(e)}")
        return "unknown" 