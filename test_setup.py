#!/usr/bin/env python3
"""
Test script to verify the English Accent Evaluator setup.
Run this script to check if all dependencies are properly installed.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'streamlit',
        'yt_dlp',
        'pydub',
        'whisper',
        'librosa',
        'numpy',
        'sklearn',
        'requests'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_ffmpeg():
    """Test if FFmpeg is available."""
    print("\n🎬 Testing FFmpeg availability...")
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        return False

def test_whisper_model():
    """Test if Whisper model can be loaded."""
    print("\n🤖 Testing Whisper model loading...")
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 English Accent Evaluator - Setup Test")
    print("=" * 50)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8 or higher is recommended")
    
    # Test imports
    failed_imports = test_imports()
    
    # Test FFmpeg
    ffmpeg_ok = test_ffmpeg()
    
    # Test Whisper
    whisper_ok = test_whisper_model()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    if not failed_imports and ffmpeg_ok and whisper_ok:
        print("🎉 All tests passed! Your setup is ready.")
        print("🚀 Run 'streamlit run app.py' to start the application.")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        
        if failed_imports:
            print(f"\n📦 Missing packages: {', '.join(failed_imports)}")
            print("   Run: pip install -r requirements.txt")
        
        if not ffmpeg_ok:
            print("\n🎬 FFmpeg is required for audio processing")
        
        if not whisper_ok:
            print("\n🤖 Whisper model loading failed")

if __name__ == "__main__":
    main() 