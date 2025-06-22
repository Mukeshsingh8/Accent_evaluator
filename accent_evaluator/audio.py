"""Audio extraction and feature engineering logic."""

from typing import Dict, Tuple
import os
import tempfile
import yt_dlp
import librosa
import numpy as np
import time

from .utils import (
    get_logger, validate_url, validate_audio_duration, 
    cleanup_temp_files, generate_request_id, sanitize_filename
)
from .config import ERROR_MESSAGES, WHISPER_MODEL

logger = get_logger("audio")

def extract_audio_from_video(url: str) -> Tuple[str, str]:
    """
    Download video from URL and extract audio as a .wav file.
    Returns (audio_file_path, request_id).
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting audio extraction from URL: {url[:50]}...")
    
    # Validate URL
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        logger.error(f"[{request_id}] Invalid URL: {error_msg}")
        raise ValueError(error_msg)
    
    temp_dir = None
    
    # Try multiple user agents if the first one fails
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]
    
    for attempt, user_agent in enumerate(user_agents, 1):
        try:
            temp_dir = tempfile.mkdtemp()
            logger.debug(f"[{request_id}] Created temp directory: {temp_dir}")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': True,
                'no_warnings': True,
                # Add user agent to avoid 403 errors
                'http_headers': {
                    'User-Agent': user_agent
                },
                # Add retry logic
                'retries': 5,
                'fragment_retries': 5,
                # Add cookies file if available
                'cookiefile': None,
                # Add more robust error handling
                'ignoreerrors': False,
                'no_check_certificate': True,
                # Add sleep between requests
                'sleep_interval': 2,
                'max_sleep_interval': 10,
                # Add more options for better compatibility
                'extractor_retries': 3,
                'socket_timeout': 30,
            }
            
            start_time = time.time()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"[{request_id}] Downloading video (attempt {attempt}/{len(user_agents)})...")
                info = ydl.extract_info(url, download=True)
                
                # Validate duration
                duration = info.get('duration', 0)
                is_valid_duration, duration_error = validate_audio_duration(duration)
                if not is_valid_duration:
                    logger.error(f"[{request_id}] Duration validation failed: {duration_error}")
                    raise ValueError(duration_error)
                
                title = sanitize_filename(info.get('title', 'video'))
                audio_file = os.path.join(temp_dir, f"{title}.wav")
                
                if os.path.exists(audio_file):
                    logger.info(f"[{request_id}] Audio extraction completed in {time.time() - start_time:.2f}s")
                    return audio_file, request_id
                
                # Try to find the wav file
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav'):
                        audio_file = os.path.join(temp_dir, file)
                        logger.info(f"[{request_id}] Audio extraction completed in {time.time() - start_time:.2f}s")
                        return audio_file, request_id
                
                raise Exception(ERROR_MESSAGES["audio_extraction_failed"])
                
        except Exception as e:
            logger.warning(f"[{request_id}] Attempt {attempt} failed: {str(e)}")
            # Cleanup temp directory on error
            if temp_dir and os.path.exists(temp_dir):
                cleanup_temp_files(temp_dir)
            
            # If this was the last attempt, raise the error
            if attempt == len(user_agents):
                logger.error(f"[{request_id}] All download attempts failed")
                raise Exception(f"Error downloading video after {len(user_agents)} attempts: {str(e)}")
            
            # Wait before next attempt
            time.sleep(2)
    
    # This should never be reached, but just in case
    raise Exception("Unexpected error in audio extraction")

def extract_audio_features(audio_file: str, request_id: str = None) -> Dict:
    """
    Extract audio features for accent analysis from a .wav file.
    Returns a dictionary of features.
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Starting audio feature extraction")
    
    try:
        start_time = time.time()
        
        # Load audio file
        logger.debug(f"[{request_id}] Loading audio file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract features
        logger.debug(f"[{request_id}] Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        logger.debug(f"[{request_id}] Extracting spectral features...")
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        logger.debug(f"[{request_id}] Extracting pitch features...")
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > 0.1])
        
        logger.debug(f"[{request_id}] Extracting rhythm features...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Calculate duration
        duration = len(y) / sr
        
        features = {
            'mfcc_mean': mfcc_mean.tolist(),  # Convert numpy arrays to lists for JSON serialization
            'mfcc_std': mfcc_std.tolist(),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'pitch_mean': float(pitch_mean),
            'tempo': float(tempo),
            'duration': float(duration),
            'sample_rate': int(sr)
        }
        
        logger.info(f"[{request_id}] Audio feature extraction completed in {time.time() - start_time:.2f}s")
        return features
        
    except Exception as e:
        logger.error(f"[{request_id}] Audio feature extraction failed: {str(e)}")
        raise Exception(f"Error extracting audio features: {str(e)}")

def cleanup_audio_file(audio_file: str, request_id: str = None) -> None:
    """Cleanup audio file after processing."""
    if request_id is None:
        request_id = generate_request_id()
    
    logger.debug(f"[{request_id}] Cleaning up audio file: {audio_file}")
    cleanup_temp_files(audio_file) 