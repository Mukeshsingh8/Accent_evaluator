"""LLM-based accent analysis using OpenAI GPT-4."""
from typing import Dict, Optional
import openai
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from .utils import (
    get_logger, validate_api_key, safe_json_loads, 
    generate_request_id, format_error_response
)
from .config import OPENAI_MODEL, LLM_PROMPT_TEMPLATE, ERROR_MESSAGES

logger = get_logger("llm")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_accent_analysis(transcript: str, audio_features: Dict, openai_api_key: str, request_id: str = None) -> Dict:
    """
    Use OpenAI GPT-4 to analyze accent from transcript and audio features.
    Returns a dict with keys: accent, confidence, explanation.
    """
    if request_id is None:
        request_id = generate_request_id()
    
    logger.info(f"[{request_id}] Starting LLM accent analysis")
    
    # Validate API key
    is_valid, error_msg = validate_api_key(openai_api_key)
    if not is_valid:
        logger.error(f"[{request_id}] Invalid API key: {error_msg}")
        raise ValueError(error_msg)
    
    try:
        start_time = time.time()
        
        # Prepare prompt
        prompt = LLM_PROMPT_TEMPLATE.format(
            transcript=transcript,
            audio_features=json.dumps(audio_features, indent=2)
        )
        
        logger.debug(f"[{request_id}] Sending request to OpenAI API")
        
        # Make API call using the correct format for openai library v1.3.8
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        result = safe_json_loads(content)
        if result is None:
            logger.warning(f"[{request_id}] Failed to parse JSON response, using fallback")
            return {
                "accent": "Unknown",
                "confidence": 0,
                "explanation": f"Failed to parse LLM response: {content[:200]}..."
            }
        
        # Validate response structure
        required_keys = ["accent", "confidence", "explanation"]
        if not all(key in result for key in required_keys):
            logger.warning(f"[{request_id}] Invalid response structure: {result}")
            return {
                "accent": "Unknown",
                "confidence": 0,
                "explanation": "Invalid response structure from LLM"
            }
        
        # Validate confidence score
        confidence = result.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
            logger.warning(f"[{request_id}] Invalid confidence score: {confidence}")
            result["confidence"] = max(0, min(100, float(confidence) if confidence else 0))
        
        logger.info(f"[{request_id}] LLM analysis completed in {time.time() - start_time:.2f}s")
        logger.debug(f"[{request_id}] LLM result: {result}")
        
        return result
        
    except openai.AuthenticationError:
        logger.error(f"[{request_id}] OpenAI authentication failed")
        raise ValueError(ERROR_MESSAGES["invalid_api_key"])
    except openai.RateLimitError:
        logger.error(f"[{request_id}] OpenAI rate limit exceeded")
        raise ValueError(ERROR_MESSAGES["rate_limit_exceeded"])
    except openai.APIError as e:
        logger.error(f"[{request_id}] OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] LLM analysis failed: {str(e)}")
        raise Exception(f"LLM analysis failed: {str(e)}")

def validate_llm_response(response: Dict) -> bool:
    """
    Validate LLM response structure and content.
    Returns True if valid, False otherwise.
    """
    required_keys = ["accent", "confidence", "explanation"]
    
    # Check required keys
    if not all(key in response for key in required_keys):
        return False
    
    # Check accent is string
    if not isinstance(response["accent"], str):
        return False
    
    # Check confidence is numeric and in range
    try:
        confidence = float(response["confidence"])
        if confidence < 0 or confidence > 100:
            return False
    except (ValueError, TypeError):
        return False
    
    # Check explanation is string
    if not isinstance(response["explanation"], str):
        return False
    
    return True 