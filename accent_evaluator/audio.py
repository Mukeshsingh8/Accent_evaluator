"""Audio extraction and feature engineering logic."""

from typing import Dict, Tuple
import os
import tempfile
import yt_dlp
import librosa
import numpy as np
import time
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
    Extract audio from a video URL using yt-dlp.
    Returns (audio_file_path, request_id).
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting audio extraction from URL: {url}...")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"[{request_id}] Created temp directory: {temp_dir}")
        
        # Test YouTube accessibility for YouTube URLs
        if 'youtube.com' in url or 'youtu.be' in url:
            is_accessible = _test_youtube_access(request_id)
            if not is_accessible:
                raise Exception("YouTube is not accessible. Please try file upload instead.")
        
        # Single yt-dlp configuration - download video first, extract audio separately
        ydl_opts = {
            'format': 'best[height<=720]/best',  # Download best quality video
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            },
            'retries': 3,
            'fragment_retries': 3,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'sleep_interval': 2,
            'max_sleep_interval': 10,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'geo_bypass_ip_block': '1.0.0.1',
            'age_limit': 0,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                    'player_skip': ['webpage', 'configs'],
                    'player_params': {
                        'hl': 'en',
                        'gl': 'US',
                    }
                }
            },
            'cookiefile': None,
            'cookiesfrombrowser': None,
            'socket_timeout': 30,
            'extractor_retries': 3,
            'file_access_retries': 3,
            'retry_sleep': 1,
        }
        
        start_time = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"[{request_id}] Downloading video with yt-dlp...")
            result = ydl.extract_info(url, download=True)
            
            # Handle different return types from extract_info
            logger.debug(f"[{request_id}] extract_info result type: {type(result)}")
            if isinstance(result, tuple):
                logger.debug(f"[{request_id}] extract_info returned tuple with {len(result)} elements")
                if len(result) == 2:
                    info, _ = result  # Unpack tuple (info, download_path)
                elif len(result) == 1:
                    info = result[0]  # Single element tuple
                else:
                    raise Exception(f"Unexpected tuple length from extract_info: {len(result)}")
            else:
                info = result  # Direct dictionary
            
            logger.debug(f"[{request_id}] Info type: {type(info)}")
            
            # Validate duration
            duration = info.get('duration', 0)
            is_valid_duration, duration_error = validate_audio_duration(duration)
            if not is_valid_duration:
                logger.error(f"[{request_id}] Duration validation failed: {duration_error}")
                raise ValueError(duration_error)
            
            # Find the downloaded video file
            video_file = None
            for file in os.listdir(temp_dir):
                if not file.endswith('.wav'):  # Skip any existing wav files
                    video_file = os.path.join(temp_dir, file)
                    break
            
            if not video_file:
                raise Exception("No video file found after download")
            
            logger.info(f"[{request_id}] Video downloaded: {video_file}")
            
            # Extract audio using FFmpeg
            title = sanitize_filename(info.get('title', 'video'))
            audio_file = os.path.join(temp_dir, f"{title}.wav")
            
            import subprocess
            cmd = [
                'ffmpeg', '-i', video_file,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                audio_file
            ]
            
            logger.info(f"[{request_id}] Extracting audio with FFmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"[{request_id}] FFmpeg failed: {result.stderr}")
                raise Exception(f"Failed to extract audio: {result.stderr}")
            
            if os.path.exists(audio_file):
                logger.info(f"[{request_id}] Audio extraction completed in {time.time() - start_time:.2f}s")
                return audio_file, request_id
            
            # Try to find the wav file if the expected name doesn't exist
            for file in os.listdir(temp_dir):
                if file.endswith('.wav'):
                    audio_file = os.path.join(temp_dir, file)
                    logger.info(f"[{request_id}] Audio extraction completed in {time.time() - start_time:.2f}s")
                    return audio_file, request_id
            
            raise Exception(ERROR_MESSAGES["audio_extraction_failed"])
            
    except Exception as e:
        logger.error(f"[{request_id}] Audio extraction failed: {str(e)}")
        # Cleanup temp directory on error
        if temp_dir and os.path.exists(temp_dir):
            cleanup_temp_files(temp_dir)
        raise Exception(f"Error downloading video: {str(e)}")

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
    Process an uploaded video or audio file and extract audio.
    Supports video files (MP4, AVI, MOV, etc.) and audio files (MP3, WAV, M4A, etc.).
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
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"[{request_id}] File saved to: {file_path}")
        
        # Determine file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Audio file extensions that can be processed directly
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma'}
        
        # Video file extensions that need audio extraction
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
        
        if file_extension in audio_extensions:
            # Audio file - process directly
            logger.info(f"[{request_id}] Processing audio file directly: {file_extension}")
            audio_file_path = file_path
            
            # For non-WAV audio files, convert to WAV for consistency
            if file_extension != '.wav':
                wav_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")
                import subprocess
                cmd = [
                    'ffmpeg', '-i', file_path,
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', '16000',  # Sample rate
                    '-ac', '1',  # Mono
                    '-y',  # Overwrite output
                    wav_file_path
                ]
                
                logger.info(f"[{request_id}] Converting audio to WAV format...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.error(f"[{request_id}] FFmpeg conversion failed: {result.stderr}")
                    raise Exception(f"Failed to convert audio: {result.stderr}")
                
                audio_file_path = wav_file_path
                
        elif file_extension in video_extensions:
            # Video file - extract audio
            logger.info(f"[{request_id}] Extracting audio from video file: {file_extension}")
            audio_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")
            
            import subprocess
            cmd = [
                'ffmpeg', '-i', file_path,
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
        else:
            raise Exception(f"Unsupported file type: {file_extension}. Supported formats: {', '.join(audio_extensions | video_extensions)}")
        
        if not os.path.exists(audio_file_path):
            raise Exception("Audio file was not created")
        
        # Validate audio file
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            duration = len(y) / sr
            logger.info(f"[{request_id}] Audio processed successfully: {duration:.2f}s duration")
            
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