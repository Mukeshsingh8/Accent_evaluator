"""Utility/helper functions for accent evaluator."""
import logging
import logging.config
import os
import re
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import hashlib
import json
from datetime import datetime, timedelta
from collections import defaultdict
import threading

from .config import LOGGING_CONFIG, ERROR_MESSAGES, MAX_AUDIO_DURATION, SUPPORTED_VIDEO_FORMATS

# Rate limiting storage
_request_counts = defaultdict(list)
_rate_limit_lock = threading.Lock()

def setup_logging() -> None:
    """Setup logging configuration."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"accent_evaluator.{name}")

def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate if the URL is supported.
    Returns (is_valid, error_message).
    """
    if not url or not url.strip():
        return False, ERROR_MESSAGES["invalid_url"]
    
    url = url.strip()
    
    # Check if it's a valid URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, ERROR_MESSAGES["invalid_url"]
    except Exception:
        return False, ERROR_MESSAGES["invalid_url"]
    
    # Check if it's a supported platform
    supported_domains = [
        "youtube.com", "youtu.be", "www.youtube.com",
        "loom.com", "www.loom.com",
        "vimeo.com", "www.vimeo.com"
    ]
    
    domain = parsed.netloc.lower()
    if not any(supported in domain for supported in supported_domains):
        # Allow direct file URLs
        if not url.lower().endswith(tuple(SUPPORTED_VIDEO_FORMATS)):
            return False, ERROR_MESSAGES["unsupported_format"]
    
    return True, ""

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate OpenAI API key format.
    Returns (is_valid, error_message).
    """
    if not api_key or not api_key.strip():
        return False, ERROR_MESSAGES["invalid_api_key"]
    
    # Basic OpenAI API key validation (starts with sk- and has proper length)
    if not api_key.startswith("sk-") or len(api_key) < 20:
        return False, ERROR_MESSAGES["invalid_api_key"]
    
    return True, ""

def check_rate_limit(user_id: str, limit_type: str = "minute") -> Tuple[bool, str]:
    """
    Check if user has exceeded rate limits.
    Returns (allowed, error_message).
    """
    with _rate_limit_lock:
        current_time = time.time()
        user_requests = _request_counts[user_id]
        
        # Clean old requests
        if limit_type == "minute":
            cutoff_time = current_time - 60
            limit = 10
        else:  # hour
            cutoff_time = current_time - 3600
            limit = 100
        
        user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff_time]
        
        if len(user_requests) >= limit:
            return False, ERROR_MESSAGES["rate_limit_exceeded"]
        
        user_requests.append(current_time)
        return True, ""

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.md5(f"{timestamp}_{time.time()}".encode()).hexdigest()[:8]
    return f"req_{timestamp}_{random_part}"

def validate_audio_duration(duration_seconds: float) -> Tuple[bool, str]:
    """
    Validate audio duration is within limits.
    Returns (is_valid, error_message).
    """
    if duration_seconds > MAX_AUDIO_DURATION:
        return False, ERROR_MESSAGES["file_too_large"]
    return True, ""

def safe_json_loads(data: str) -> Optional[Dict]:
    """Safely parse JSON with error handling."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None

def format_error_response(error_type: str, message: str, request_id: str) -> Dict:
    """Format error response for API consistency."""
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    }

def format_success_response(data: Dict, request_id: str) -> Dict:
    """Format success response for API consistency."""
    return {
        "success": True,
        "data": data,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat()
    }

def cleanup_temp_files(file_path: str) -> None:
    """Safely cleanup temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger = get_logger("utils")
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 0.0 