"""
AI Text Humanizer - Streamlit Web UI
Select between Balanced and Aggressive humanizers
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import humanizer_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from humanizer_core import AdvancedAITextHumanizer, UltraAggressiveHumanizer

# Page configuration
st.set_page_config(
    page_title="AI Text Humanizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'humanized_text' not in st.session_state:
    st.session_state.humanized_text = ""
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'balanced_humanizer' not in st.session_state:
    st.session_state.balanced_humanizer = None
if 'aggressive_humanizer' not in st.session_state:
    st.session_state.aggressive_humanizer = None

# Header
st.markdown('<div class="main-header">🧠 AI Text Humanizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transform AI-generated text into natural, human-like writing</div>', unsafe_allow_html=True)

# Sidebar - Humanizer Selection
with st.sidebar:
    st.header("⚙️ Settings")
    
    humanizer_type = st.radio(
        "Choose Humanizer:",
        options=["🟢 Balanced", "🔴 Aggressive"],
        help="Select the humanization style you prefer"
    )
    
    st.markdown("---")
    
    # Humanizer descriptions
    if humanizer_type == "🟢 Balanced":
        st.markdown("""
        ### 🟢 Balanced Humanizer
        
        **Best for:**
        - Academic writing
        - Professional documents
        - Formal content
        
        **Features:**
        - Maintains clarity
        - Preserves grammar
        - Moderate rephrasing
        - ~80% transformation rate
        
        **Target:** <50% AI detection
        """)
    else:
        st.markdown("""
        ### 🔴 Aggressive Humanizer
        
        **Best for:**
        - Creative content
        - Blog posts
        - Casual writing
        
        **Features:**
        - Strong rephrasing
        - Conversational tone
        - Heavy transformation
        - ~95% transformation rate
        
        **Target:** <30% AI detection
        """)
    
    st.markdown("---")
    
    # Sample texts
    st.subheader("📝 Sample Texts")
    sample_option = st.selectbox(
        "Load a sample:",
        options=[
            "None",
            "Academic Writing",
            "Business Email",
            "Tech Article"
        ]
    )
    
    samples = {
        "Academic Writing": """Furthermore, the implementation of artificial intelligence in educational settings demonstrates significant potential for enhancing learning outcomes. Moreover, various studies have indicated that personalized learning approaches facilitate improved student engagement. Therefore, it is essential to establish comprehensive frameworks for integrating these technologies effectively.""",
        
        "Business Email": """Additionally, it is important to note that our organization has implemented various strategies to optimize operational efficiency. Furthermore, we have established comprehensive protocols to ensure quality standards are maintained. Therefore, we would like to schedule a meeting to discuss these developments in detail.""",
        
        "Tech Article": """Artificial intelligence is revolutionizing numerous industries through innovative applications. Moreover, machine learning algorithms demonstrate remarkable capabilities in processing extensive datasets. Consequently, organizations are increasingly utilizing these technologies to enhance their operational frameworks and establish competitive advantages."""
    }

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    
    # Load sample if selected
    default_text = ""
    if sample_option != "None":
        default_text = samples[sample_option]
    
    input_text = st.text_area(
        "Paste your AI-generated text here:",
        value=default_text,
        height=300,
        placeholder="Enter or paste the text you want to humanize...",
        help="The text will be processed to sound more natural and human-like"
    )
    
    # Character count
    char_count = len(input_text)
    word_count = len(input_text.split())
    st.caption(f"📊 {char_count} characters | {word_count} words")
    
    # Upload file option
    with st.expander("📎 Or upload a text file"):
        uploaded_file = st.file_uploader(
            "Choose a .txt file",
            type=['txt'],
            help="Upload a plain text file to humanize"
        )
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode('utf-8')
            st.success(f"✅ Loaded {len(input_text)} characters from file")

with col2:
    st.subheader("📤 Humanized Output")
    
    # Humanize button
    if st.button("🚀 Humanize Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("⚠️ Please enter some text to humanize!")
        else:
            with st.spinner(f"Processing with {humanizer_type} humanizer..."):
                try:
                    # Initialize humanizer based on selection (only once)
                    if humanizer_type == "🟢 Balanced":
                        if st.session_state.balanced_humanizer is None:
                            with st.spinner("Loading Balanced humanizer (first time only)..."):
                                st.session_state.balanced_humanizer = AdvancedAITextHumanizer(load_datasets=True)
                        
                        humanized, stats = st.session_state.balanced_humanizer.humanize(input_text, intensity='maximum')
                    else:
                        if st.session_state.aggressive_humanizer is None:
                            with st.spinner("Loading Aggressive humanizer (first time only)..."):
                                st.session_state.aggressive_humanizer = UltraAggressiveHumanizer(load_datasets=True)
                        
                        humanized, stats = st.session_state.aggressive_humanizer.humanize(input_text, intensity='ultra')
                    
                    st.session_state.humanized_text = humanized
                    st.session_state.stats = stats
                    st.success("✅ Text humanized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure all required packages are installed. Run: pip install -r requirements.txt")
    
    # Display output
    if st.session_state.humanized_text:
        output_text = st.text_area(
            "Your humanized text:",
            value=st.session_state.humanized_text,
            height=300,
            help="Copy this text or download it below"
        )
        
        # Download button
        st.download_button(
            label="💾 Download Humanized Text",
            data=st.session_state.humanized_text,
            file_name="humanized_text.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Copy button (using streamlit's clipboard)
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.code(st.session_state.humanized_text, language=None)
            st.info("💡 Select the text above and press Ctrl+C (or Cmd+C on Mac) to copy")

# Statistics section
if st.session_state.stats:
    st.markdown("---")
    st.subheader("📊 Transformation Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Initial AI Score",
            value=f"{st.session_state.stats['initial_ai_score']:.2f}",
            help="Higher score = more AI-like"
        )
    
    with col2:
        st.metric(
            label="Final AI Score",
            value=f"{st.session_state.stats['final_ai_score']:.2f}",
            delta=f"-{st.session_state.stats['initial_ai_score'] - st.session_state.stats['final_ai_score']:.2f}",
            delta_color="inverse",
            help="Lower is better"
        )
    
    with col3:
        st.metric(
            label="Improvement",
            value=f"{st.session_state.stats['improvement']:.1f}%",
            help="Percentage reduction in AI-like patterns"
        )
    
    with col4:
        target_status = "✅ Yes" if st.session_state.stats['target_achieved'] else "❌ No"
        st.metric(
            label="Target Achieved",
            value=target_status,
            help="Whether the humanization goal was met"
        )
    
    # Progress bar for improvement
    st.progress(min(st.session_state.stats['improvement'] / 100, 1.0))
    
    if st.session_state.stats['target_achieved']:
        st.success("🎯 Great! Your text should now pass most AI detectors.")
    else:
        st.warning("⚠️ Consider running the text through the humanizer again for better results.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧠 AI Text Humanizer | Built with Streamlit</p>
        <p style='font-size: 0.9rem;'>Transform AI-generated content into natural, human-like writing</p>
    </div>
""", unsafe_allow_html=True)

# Instructions in expander
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Instructions:
    
    1. **Choose a Humanizer** in the sidebar:
       - 🟢 **Balanced**: Best for formal/professional content
       - 🔴 **Aggressive**: Best for casual/creative content
    
    2. **Input your text**:
       - Paste directly into the text area
       - Or upload a .txt file
       - Or select a sample text
    
    3. **Click "Humanize Text"** and wait for processing
    
    4. **Review the output**:
       - Check the statistics to see improvement
       - Copy or download the humanized text
    
    5. **Run again** if needed for even better results
    
    ### Tips:
    - Longer texts may take more time to process
    - The Aggressive humanizer is more thorough but takes slightly longer
    - You can run the same text multiple times for cumulative effects
    """)
