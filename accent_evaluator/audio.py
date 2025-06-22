"""Audio extraction and feature engineering logic."""

from typing import Dict, Tuple
import os
import tempfile
import yt_dlp
import librosa
import numpy as np
import time
import re
import requests

from .utils import (
    get_logger, validate_url, validate_audio_duration, 
    cleanup_temp_files, generate_request_id, sanitize_filename
)
from .config import ERROR_MESSAGES, WHISPER_MODEL

logger = get_logger("audio")

def _test_youtube_access(request_id: str) -> bool:
    """Test if YouTube is accessible with a simple request."""
    test_url = "https://www.youtube.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        logger.info(f"[{request_id}] YouTube accessibility test: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"[{request_id}] YouTube accessibility test failed: {str(e)}")
        return False

def extract_audio_from_video(url: str) -> Tuple[str, str]:
    """
    Download video from URL and extract audio as a .wav file.
    Returns (audio_file_path, request_id).
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting audio extraction from URL: {url[:50]}...")
    
    # Test YouTube accessibility first
    if 'youtube.com' in url or 'youtu.be' in url:
        if not _test_youtube_access(request_id):
            logger.warning(f"[{request_id}] YouTube appears to be blocked or inaccessible")
    
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
                    'User-Agent': user_agent,
                },
                # Add retry logic
                'retries': 5,
                'fragment_retries': 5,
                # Add more robust error handling
                'ignoreerrors': False,
                'no_check_certificate': True,
                # Add sleep between requests
                'sleep_interval': 2,
                'max_sleep_interval': 10,
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
            
            # If this was the last attempt, try the fallback method
            if attempt == len(user_agents):
                logger.info(f"[{request_id}] Trying fallback method with different URL formats...")
                try:
                    temp_dir = tempfile.mkdtemp()
                    # Only try YouTube formats if it's actually a YouTube URL
                    if 'youtube.com' in url or 'youtu.be' in url:
                        audio_file = _try_different_youtube_formats(url, temp_dir, user_agents[0], request_id)
                    else:
                        # For non-YouTube URLs, just try the original URL with simpler options
                        audio_file = _try_simple_download(url, temp_dir, user_agents[0], request_id)
                    logger.info(f"[{request_id}] Fallback method succeeded")
                    return audio_file, request_id
                except Exception as fallback_error:
                    logger.error(f"[{request_id}] Fallback method also failed: {str(fallback_error)}")
                    if temp_dir and os.path.exists(temp_dir):
                        cleanup_temp_files(temp_dir)
                    raise Exception(f"Error downloading video after {len(user_agents)} attempts and fallback: {str(e)}")
            
            # Wait before next attempt
            time.sleep(3)
    
    # This should never be reached, but just in case
    raise Exception("Unexpected error in audio extraction")

def _try_simple_download(url: str, temp_dir: str, user_agent: str, request_id: str) -> str:
    """Try simple download for non-YouTube URLs."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': user_agent,
        },
        'retries': 3,
        'fragment_retries': 3,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logger.info(f"[{request_id}] Trying simple download: {url}")
        info = ydl.extract_info(url, download=True)
        
        # Check for wav file
        for file in os.listdir(temp_dir):
            if file.endswith('.wav'):
                return os.path.join(temp_dir, file)
    
    raise Exception("Simple download failed")

def _try_different_youtube_formats(url: str, temp_dir: str, user_agent: str, request_id: str) -> str:
    """Try different YouTube URL formats if the original fails."""
    
    # Extract video ID from URL
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)', url)
    if not video_id_match:
        raise Exception("Could not extract video ID from URL")
    
    video_id = video_id_match.group(1)
    
    # Try different URL formats
    url_formats = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://youtu.be/{video_id}",
        f"https://m.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/embed/{video_id}",
    ]
    
    for format_url in url_formats:
        try:
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': True,
                'no_warnings': True,
                'http_headers': {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Connection': 'keep-alive',
                },
                'retries': 5,
                'fragment_retries': 5,
                'nocheckcertificate': True,
                'prefer_insecure': True,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"[{request_id}] Trying format: {format_url}")
                info = ydl.extract_info(format_url, download=True)
                
                # Check for wav file
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav'):
                        return os.path.join(temp_dir, file)
                        
        except Exception as e:
            logger.warning(f"[{request_id}] Format {format_url} failed: {str(e)}")
            continue
    
    raise Exception("All YouTube URL formats failed")

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

def process_uploaded_file(uploaded_file, request_id: str = None) -> Tuple[str, str]:
    """
    Process an uploaded video file and extract audio.
    Returns (audio_file_path, request_id).
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Processing uploaded file: {uploaded_file.name}")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"[{request_id}] Created temp directory: {temp_dir}")
        
        # Save uploaded file to temp directory
        video_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"[{request_id}] File saved to: {video_file_path}")
        
        # Extract audio using FFmpeg
        audio_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")
        
        import subprocess
        cmd = [
            'ffmpeg', '-i', video_file_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            audio_file_path
        ]
        
        logger.info(f"[{request_id}] Extracting audio with FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"[{request_id}] FFmpeg failed: {result.stderr}")
            raise Exception(f"Failed to extract audio: {result.stderr}")
        
        if not os.path.exists(audio_file_path):
            raise Exception("Audio file was not created")
        
        # Validate audio file
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            duration = len(y) / sr
            logger.info(f"[{request_id}] Audio extracted successfully: {duration:.2f}s duration")
            
            # Validate duration
            is_valid_duration, duration_error = validate_audio_duration(duration)
            if not is_valid_duration:
                logger.error(f"[{request_id}] Duration validation failed: {duration_error}")
                raise ValueError(duration_error)
                
        except Exception as e:
            logger.error(f"[{request_id}] Audio validation failed: {str(e)}")
            raise Exception(f"Invalid audio file: {str(e)}")
        
        logger.info(f"[{request_id}] File processing completed successfully")
        return audio_file_path, request_id
        
    except Exception as e:
        logger.error(f"[{request_id}] File processing failed: {str(e)}")
        # Cleanup temp directory on error
        if temp_dir and os.path.exists(temp_dir):
            cleanup_temp_files(temp_dir)
        raise Exception(f"Error processing uploaded file: {str(e)}") 