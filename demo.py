#!/usr/bin/env python3
"""
Demo script for the English Accent Evaluator.
This script demonstrates how to use the AccentEvaluator class programmatically.
"""

import os
import tempfile
from app import AccentEvaluator

def demo_with_sample_audio():
    """Demo using a sample audio file (if available)."""
    print("🎤 English Accent Evaluator - Demo")
    print("=" * 50)
    
    # Initialize the evaluator
    print("🔧 Initializing AccentEvaluator...")
    evaluator = AccentEvaluator()
    print("✅ Evaluator initialized successfully")
    
    # Example usage with a sample URL
    sample_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    
    print(f"\n📹 Testing with sample URL: {sample_url}")
    print("⚠️  Note: This is a demo. Replace with a real video URL for actual analysis.")
    
    try:
        # This would normally process a real video
        # For demo purposes, we'll show the structure
        
        print("\n📋 Processing steps:")
        print("1. Download video and extract audio")
        print("2. Transcribe audio using Whisper")
        print("3. Extract audio features")
        print("4. Analyze accent patterns")
        print("5. Generate results")
        
        print("\n🎯 Supported accents:")
        for accent in evaluator.accent_keywords.keys():
            print(f"   • {accent}")
        
        print("\n📊 Example output format:")
        print("   - Detected Accent: American")
        print("   - Confidence Score: 85.5%")
        print("   - Summary: Detailed explanation...")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

def demo_accent_analysis():
    """Demo the accent analysis logic with sample data."""
    print("\n🔬 Demo: Accent Analysis Logic")
    print("-" * 30)
    
    evaluator = AccentEvaluator()
    
    # Sample transcription
    sample_transcription = "Hello, I'm from the United States. I love water and better weather."
    
    # Sample audio features
    sample_features = {
        'spectral_centroid_mean': 2100,
        'pitch_mean': 140,
        'tempo': 110
    }
    
    print(f"📝 Sample transcription: {sample_transcription}")
    print(f"🎵 Sample audio features: {sample_features}")
    
    # Analyze accent
    detected_accent, confidence, all_scores = evaluator.analyze_accent(
        sample_transcription, sample_features
    )
    
    print(f"\n🎯 Analysis Results:")
    print(f"   Detected Accent: {detected_accent}")
    print(f"   Confidence: {confidence:.1f}%")
    
    print(f"\n📊 All Accent Scores:")
    for accent, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {accent}: {score:.1f}%")
    
    # Generate summary
    summary = evaluator.generate_summary(detected_accent, confidence, sample_transcription)
    print(f"\n📝 Summary: {summary}")

def main():
    """Run the demo."""
    try:
        demo_with_sample_audio()
        demo_accent_analysis()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\n🚀 To run the full application:")
        print("   streamlit run app.py")
        
        print("\n🧪 To test your setup:")
        print("   python test_setup.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 