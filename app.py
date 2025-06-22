import streamlit as st
import time
from accent_evaluator.audio import extract_audio_from_video, extract_audio_features, cleanup_audio_file
from accent_evaluator.transcription import transcribe_audio
from accent_evaluator.llm import llm_accent_analysis
from accent_evaluator.utils import setup_logging, get_logger, check_rate_limit, generate_request_id

# Setup logging
setup_logging()
logger = get_logger("app")

# Page configuration
st.set_page_config(
    page_title="English Accent Evaluator",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 20px;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .error-box {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .progress-step {
        display: inline-block;
        margin: 0 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        opacity: 0.5;
        transform: scale(0.9);
    }
    .progress-step.active {
        opacity: 1;
        transform: scale(1.1);
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
    }
    .progress-step.completed {
        opacity: 1;
        transform: scale(1);
        background: rgba(40, 167, 69, 0.8);
    }
    .progress-bar {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
        border-radius: 4px;
        transition: width 0.5s ease;
        width: 0%;
    }
    .video-preview {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .reset-button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .reset-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def create_progress_animation(current_step: int, total_steps: int = 4):
    """Create a dynamic progress animation."""
    steps = ["📥 Download", "🎵 Transcribe", "🔬 Analyze", "🤖 AI Detect"]
    progress_percentage = (current_step / total_steps) * 100
    
    step_html = ""
    for i, step in enumerate(steps):
        if i < current_step - 1:
            status = "completed"
            icon = "✅"
        elif i == current_step - 1:
            status = "active"
            icon = "🔄"
        else:
            status = "pending"
            icon = "⏳"
        
        step_html += f'<span class="progress-step {status}">{icon} {step}</span>'
    
    st.markdown(f"""
    <div class="progress-container">
        <h3>🎤 Processing Your Video</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_percentage}%"></div>
        </div>
        <div style="margin-top: 1rem;">
            {step_html}
        </div>
        <p style="margin-top: 1rem; opacity: 0.9;">Step {current_step} of {total_steps}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎤 English Accent Evaluator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Analyze English accents in video interviews using advanced AI technology
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Generate request ID for tracking
    request_id = generate_request_id()
    
    # Rate limiting (simple IP-based)
    user_ip = st.experimental_get_query_params().get("user_ip", ["unknown"])[0]
    rate_allowed, rate_error = check_rate_limit(user_ip)
    if not rate_allowed:
        st.error(f"⚠️ {rate_error}")
        return

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("**Analysis Method:** OpenAI GPT-4")
        st.info("Using advanced AI for accurate accent detection")
        
        st.markdown("### 📊 Statistics")
        if 'total_analyses' not in st.session_state:
            st.session_state.total_analyses = 0
        st.metric("Total Analyses", st.session_state.total_analyses)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Video URL Input")
        video_url = st.text_input(
            "Enter video URL (YouTube, Loom, or direct MP4):",
            placeholder="https://www.youtube.com/watch?v=... or https://www.loom.com/share/..."
        )
        
        # OpenAI API Key input
        openai_api_key = st.text_input("Enter your OpenAI API key:", type="password", help="Required for GPT-4 accent analysis")
        
        if st.button("🔍 Analyze Accent", type="primary", use_container_width=True):
            if not video_url:
                st.error("Please enter a valid video URL.")
                return
            if not openai_api_key:
                st.error("Please enter your OpenAI API key.")
                return
            
            # Start processing
            start_time = time.time()
            audio_file = None
            
            try:
                logger.info(f"[{request_id}] Starting accent analysis for user {user_ip}")
                
                # Progress animation placeholder
                progress_placeholder = st.empty()
                
                # Step 1: Extract audio
                with progress_placeholder.container():
                    create_progress_animation(1)
                
                with st.spinner("📥 Downloading video and extracting audio..."):
                    audio_file, audio_request_id = extract_audio_from_video(video_url)
                
                # Step 2: Transcribe audio
                with progress_placeholder.container():
                    create_progress_animation(2)
                
                with st.spinner("🎵 Transcribing audio with Whisper..."):
                    transcription = transcribe_audio(audio_file, audio_request_id)
                
                # Step 3: Extract audio features
                with progress_placeholder.container():
                    create_progress_animation(3)
                
                with st.spinner("🔬 Analyzing audio features..."):
                    audio_features = extract_audio_features(audio_file, audio_request_id)
                
                # Step 4: LLM Analysis
                with progress_placeholder.container():
                    create_progress_animation(4)
                
                with st.spinner("🤖 Analyzing accent with GPT-4..."):
                    llm_result = llm_accent_analysis(transcription, audio_features, openai_api_key, audio_request_id)
                
                # Clear progress animation
                progress_placeholder.empty()

                # Calculate processing time
                processing_time = time.time() - start_time
                logger.info(f"[{request_id}] Analysis completed in {processing_time:.2f}s")
                
                # Update statistics
                st.session_state.total_analyses += 1

                # Display results
                st.success(f"✅ Analysis complete! (Processed in {processing_time:.1f}s)")
                
                # Results section
                st.markdown("### 📊 Analysis Results")
                
                detected_accent = llm_result.get("accent", "Unknown")
                confidence = llm_result.get("confidence", 0)
                summary = llm_result.get("explanation", "No explanation provided.")
                
                # Stats grid
                st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3>🎯 Accent</h3>
                        <h2>{detected_accent}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3>📈 Confidence</h3>
                        <h2>{confidence:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3>⏱️ Duration</h3>
                        <h2>{processing_time:.1f}s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown("### 📈 Confidence Score")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%"></div>
                </div>
                <p style="text-align: center;"><strong>{confidence:.1f}%</strong></p>
                """, unsafe_allow_html=True)
                
                # Summary
                st.markdown("### 📝 AI Analysis Summary")
                st.info(summary)
                
                # Transcription
                with st.expander("📝 View Transcription"):
                    st.text_area("Transcribed Text:", transcription, height=150)
                
                # Request ID for support
                st.markdown(f"**Request ID:** `{request_id}`")
                
                # Reset button - make it more prominent
                st.markdown("---")
                st.markdown("### 🔄 Ready for Another Analysis?")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 Analyze Another Video", type="primary", use_container_width=True, help="Click to start a new analysis"):
                        st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{request_id}] Analysis failed: {error_msg}")
                
                # Clear progress and show error
                progress_placeholder.empty()
                
                st.error(f"❌ Error during analysis: {error_msg}")
                st.markdown(f"""
                <div class="error-box">
                    <strong>Troubleshooting:</strong><br>
                    • Check that the URL is valid and accessible<br>
                    • Ensure the video contains clear English speech<br>
                    • Verify your OpenAI API key<br>
                    • Try a different video or shorter clip<br>
                    <br>
                    <strong>Request ID:</strong> {request_id}
                </div>
                """, unsafe_allow_html=True)
                
                # Reset button on error - make it more prominent
                st.markdown("---")
                st.markdown("### 🔄 Try Again?")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 Try Again", type="primary", use_container_width=True, help="Click to try the analysis again"):
                        st.rerun()
            
            finally:
                # Cleanup temporary files
                if audio_file:
                    try:
                        cleanup_audio_file(audio_file, request_id)
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to cleanup audio file: {e}")
    
    with col2:
        st.markdown("### 🎥 Video Preview")
        if video_url:
            try:
                # Extract video ID for YouTube
                if "youtube.com" in video_url or "youtu.be" in video_url:
                    if "youtube.com/watch?v=" in video_url:
                        video_id = video_url.split("watch?v=")[1].split("&")[0]
                    elif "youtu.be/" in video_url:
                        video_id = video_url.split("youtu.be/")[1].split("?")[0]
                    else:
                        video_id = None
                    
                    if video_id:
                        st.markdown(f"""
                        <div class="video-preview">
                            <iframe 
                                width="100%" 
                                height="200" 
                                src="https://www.youtube.com/embed/{video_id}" 
                                frameborder="0" 
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen>
                            </iframe>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Video preview available for YouTube videos")
            except:
                st.info("Video preview not available")
        else:
            st.info("Enter a video URL to see preview")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use videos with clear English speech
        - Longer videos provide better analysis
        - Avoid videos with background music
        - Ensure good audio quality
        """)

if __name__ == "__main__":
    main() 