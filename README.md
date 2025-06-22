# 🎤 English Accent Evaluator

A Streamlit application that analyzes video interviews to detect English accents using OpenAI GPT-4.

## ✨ Features

- **Video Processing**: Supports Loom, Vimeo, direct MP4 URLs, and file uploads
- **Audio Support**: Direct MP3, WAV, M4A file uploads
- **Speech Recognition**: OpenAI Whisper for accurate transcription
- **AI-Powered Analysis**: GPT-4 for accurate accent detection
- **Modern UI**: Clean Streamlit interface with real-time feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mukeshsingh8/Accent_evaluator.git
   cd Accent_evaluator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Deploy

**Note**: YouTube URLs are blocked on cloud servers. Use file upload or run locally for YouTube support.

## 📊 Usage

1. **Upload File** (Recommended): Upload video/audio files directly
2. **Enter URL**: Use Loom, Vimeo, or direct MP4 URLs
3. **Enter OpenAI API Key**: Required for analysis
4. **Analyze**: Click analyze and view results

## 🔑 OpenAI API Key

1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up/Login and go to "API Keys"
3. Create a new secret key
4. Paste it in the app

**Security**: Your API key is never stored and only used for analysis sessions.

## 🏠 Local vs Cloud

| Feature | Cloud | Local |
|---------|-------|-------|
| Loom Videos | ✅ | ✅ |
| Vimeo Videos | ✅ | ✅ |
| File Uploads | ✅ | ✅ |
| YouTube Videos | ❌ | ✅ |
| Processing Speed | Slower | Faster |
| Privacy | Shared | Private |

## 📁 Supported Formats

**Video**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
**Audio**: MP3, WAV, M4A, AAC, FLAC, OGG

## 🏗️ Project Structure

```
accent_evaluator/
├── accent_evaluator/     # Core package
│   ├── audio.py         # Audio extraction
│   ├── transcription.py # Speech-to-text
│   ├── llm.py          # GPT-4 analysis
│   ├── config.py       # Configuration
│   └── utils.py        # Utilities
├── app.py              # Streamlit frontend
└── requirements.txt    # Dependencies
```

## 📝 License

This project is open source and available under the MIT License. 