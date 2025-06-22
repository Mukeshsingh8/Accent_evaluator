"""Transcription logic using Faster Whisper."""
from typing import Optional
from faster_whisper import WhisperModel
import time

from .utils import get_logger, generate_request_id
from .config import WHISPER_MODEL, ERROR_MESSAGES

logger = get_logger("transcription")

_model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    """Get or load Whisper model with singleton pattern."""
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        start_time = time.time()
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        logger.info(f"Whisper model loaded in {time.time() - start_time:.2f}s")
    return _model

def transcribe_audio(audio_file: str, request_id: str = None) -> str:
    """
    Transcribe audio using Faster Whisper. Returns the transcription text.
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Starting audio transcription")
    
    try:
        start_time = time.time()
        model = get_model()
        
        logger.debug(f"[{request_id}] Transcribing audio file: {audio_file}")
        # The transcribe method returns a generator of segments
        segments_generator = model.transcribe(audio_file, beam_size=5)
        
        # Convert generator to list and extract text
        segments = list(segments_generator)
        transcription = " ".join([segment.text for segment in segments]).strip()
        
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
    Detect the language of the audio using Faster Whisper.
    Returns language code (e.g., 'en', 'es', 'fr').
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Detecting language")
    
    try:
        model = get_model()
        # The transcribe method returns a generator of segments
        segments_generator = model.transcribe(audio_file, beam_size=5)
        
        # Convert generator to list and get language from first segment
        segments = list(segments_generator)
        language = "unknown"
        
        if segments:
            # Get language from the first segment
            first_segment = segments[0]
            if hasattr(first_segment, 'language') and first_segment.language:
                language = first_segment.language
        
        logger.info(f"[{request_id}] Detected language: {language}")
        return language
        
    except Exception as e:
        logger.error(f"[{request_id}] Language detection failed: {str(e)}")
        return "unknown" 