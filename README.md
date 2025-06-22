# 🎤 English Accent Evaluator

A production-ready Streamlit application that analyzes video interviews to detect English accents using OpenAI GPT-4.

## ✨ Features

- **Video Processing**: Supports YouTube, Loom, Vimeo, and direct MP4 URLs
- **Audio Extraction**: Automatically extracts and processes audio from video files
- **Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- **AI-Powered Analysis**: GPT-4 for accurate accent detection
- **Production Ready**: Logging, error handling, rate limiting, and monitoring
- **Modern UI**: Clean, responsive Streamlit interface with real-time feedback

## 🎯 Supported Accents

- **American English**: Rhotic pronunciation, flap consonants
- **British English**: Non-rhotic, received pronunciation
- **Australian English**: Rising intonation, distinctive vowel sounds
- **Canadian English**: Canadian raising, cot-caught merger
- **Indian English**: Influence from Indian languages
- **Other**: General international accents

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key (for LLM analysis)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd REM
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

4. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ENVIRONMENT="development"
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🌐 Free Deployment Options

### 1. **Streamlit Cloud (Recommended - Free)**

**Step 1: Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/accent-evaluator.git
git push -u origin main
```

**Step 2: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `yourusername/accent-evaluator`
5. Set main file path: `app.py`
6. Add secrets in the advanced settings:
   ```
   OPENAI_API_KEY = your-openai-api-key
   ```
7. Click "Deploy"

**Step 3: Configure Environment Variables**
In Streamlit Cloud dashboard, go to Settings → Secrets and add:
```toml
OPENAI_API_KEY = "your-openai-api-key"
```

### 2. **Railway (Free Tier)**

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables:
   - `OPENAI_API_KEY`
4. Deploy

### 3. **Render (Free Tier)**

1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Add environment variables
5. Deploy

### 4. **Heroku (Free Tier - Limited)**

1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Environment (development/production) |
| `DEBUG` | `false` | Enable debug logging |
| `OPENAI_API_KEY` | `""` | OpenAI API key for LLM analysis |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model for accent analysis |
| `MAX_AUDIO_DURATION` | `300` | Maximum audio duration in seconds |
| `MAX_REQUESTS_PER_MINUTE` | `10` | Rate limit per minute |
| `MAX_REQUESTS_PER_HOUR` | `100` | Rate limit per hour |

## 🏗️ Architecture

```
REM/
├── accent_evaluator/          # Core package
│   ├── __init__.py           # Package initialization
│   ├── audio.py              # Audio extraction & features
│   ├── transcription.py      # Speech-to-text
│   ├── accent.py             # Rule-based accent analysis
│   ├── llm.py                # LLM-based accent analysis
│   ├── config.py             # Configuration & constants
│   └── utils.py              # Utilities & helpers
├── app.py                    # Streamlit frontend
├── requirements.txt          # Dependencies
├── packages.txt              # System dependencies
├── .streamlit/config.toml    # Streamlit configuration
├── README.md                 # Documentation
└── logs/                     # Application logs
```

## 📊 Usage

### Web Interface

1. **Enter Video URL**: Paste a YouTube, Loom, or direct MP4 URL
2. **Enter API Key**: Required for GPT-4 analysis
3. **Analyze**: Click the analyze button and watch the progress
4. **View Results**: See detected accent, confidence, and AI explanation

### Programmatic Usage

```python
from accent_evaluator import (
    extract_audio_from_video, 
    transcribe_audio, 
    llm_accent_analysis
)

# Extract audio from video
audio_file, request_id = extract_audio_from_video("https://youtube.com/watch?v=...")

# Transcribe audio
transcription = transcribe_audio(audio_file, request_id)

# LLM-based analysis
llm_result = llm_accent_analysis(transcription, audio_features, api_key, request_id)
```

## 📈 Monitoring & Logging

### Logs

Application logs are stored in `logs/accent_evaluator.log` with rotation:

- **INFO**: General application flow
- **DEBUG**: Detailed processing information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and failures

### Metrics

The application tracks:
- Request processing time
- Success/failure rates
- Rate limiting events
- Model loading times

## 🔒 Security

### API Key Management

- API keys are never stored or logged
- Keys are validated before use
- Secure input fields in the UI

### Rate Limiting

- Per-user rate limiting (IP-based)
- Configurable limits per minute/hour
- Graceful error handling

### Input Validation

- URL validation and sanitization
- File format validation
- Duration limits
- Content type verification

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```

2. **OpenAI API errors**
   - Verify API key is correct
   - Check API key has sufficient credits
   - Ensure API key has access to GPT-4

3. **Audio extraction fails**
   - Check video URL is accessible
   - Verify video contains audio
   - Try a different video format

4. **Transcription issues**
   - Ensure video contains clear English speech
   - Check audio quality
   - Try shorter video clips

### Debug Mode

Enable debug logging:

```bash
export DEBUG=true
streamlit run app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review application logs
- Contact the development team

---

**Made with ❤️ for better English accent analysis** 