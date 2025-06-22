"""Accent analysis logic."""
from typing import Dict, Tuple
from .config import ACCENT_KEYWORDS

def analyze_accent(transcription: str, audio_features: Dict) -> Tuple[str, float, Dict[str, float]]:
    """
    Analyze accent based on transcription and audio features.
    Returns (detected_accent, confidence, all_scores).
    """
    transcription_lower = transcription.lower()
    accent_scores = {}
    for accent, keywords in ACCENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in transcription_lower:
                score += 10
            elif any(word in transcription_lower for word in keyword.split()):
                score += 5
        # Audio feature analysis (heuristics)
        if accent == "American":
            if audio_features.get('spectral_centroid_mean', 0) > 2000:
                score += 15
        elif accent == "British":
            if audio_features.get('pitch_mean', 0) > 150:
                score += 15
        elif accent == "Australian":
            if audio_features.get('tempo', 0) > 120:
                score += 15
        elif accent == "Canadian":
            if 1800 < audio_features.get('spectral_centroid_mean', 0) < 2200:
                score += 15
        elif accent == "Indian":
            if audio_features.get('tempo', 0) < 100:
                score += 15
        accent_scores[accent] = score
    max_score = max(accent_scores.values()) if accent_scores else 1
    for accent in accent_scores:
        accent_scores[accent] = min(100, (accent_scores[accent] / max_score) * 100)
    detected_accent = max(accent_scores, key=accent_scores.get)
    confidence = accent_scores[detected_accent]
    return detected_accent, confidence, accent_scores

def generate_summary(detected_accent: str, confidence: float, transcription: str) -> str:
    """
    Generate a summary explanation for the detected accent.
    """
    summaries = {
        "American": "American English is characterized by rhotic pronunciation (pronouncing 'r' sounds), flapped 't' sounds in words like 'water', and specific vowel patterns.",
        "British": "British English features non-rhotic pronunciation (dropping 'r' sounds), distinctive vowel sounds, and often uses Received Pronunciation patterns.",
        "Australian": "Australian English is known for its rising intonation, distinctive vowel shifts, and unique slang expressions.",
        "Canadian": "Canadian English features Canadian raising, specific vowel mergers, and often shows influence from both American and British English.",
        "Indian": "Indian English has distinctive rhythm patterns, retroflex consonants, and often shows influence from local Indian languages."
    }
    base_summary = summaries.get(detected_accent, "The accent analysis is based on pronunciation patterns, intonation, and linguistic features.")
    if confidence > 80:
        confidence_desc = "high confidence"
    elif confidence > 60:
        confidence_desc = "moderate confidence"
    else:
        confidence_desc = "low confidence"
    return f"Detected {detected_accent} accent with {confidence_desc} ({confidence:.1f}%). {base_summary}" 