import streamlit as st
import preprocessor
from enhanced_helper_1 import *
from enhanced_helper_2 import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Enable caching for better performance
@st.cache_data
def load_data(uploaded_file):
    """Cache the data loading process"""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        return preprocessor.preprocess(data)
    return None

@st.cache_data
def get_cached_stats(selected_user, df):
    """Cache basic statistics"""
    return fetch_stats(selected_user, df)

# Set page configuration
st.set_page_config(
    page_title="Chat Analyzer - Discover Your Conversations! 🎉",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Professional modern CSS styling with excellent font visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        transition: all 0.3s ease;
    }
    
    /* Light Theme (Default) - Professional & Modern */
    .stApp:not(.dark-mode) {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #cbd5e1 50%, #94a3b8 75%, #64748b 100%);
        color: #1e293b;
    }
    
    /* Dark Theme - Professional & Modern */
    .stApp.dark-mode {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #64748b 100%);
        color: #f1f5f9;
    }
    
    /* Professional Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #3b82f6, #1d4ed8, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .dark-mode .main-header {
        background: linear-gradient(45deg, #60a5fa, #3b82f6, #1d4ed8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .dark-mode .subtitle {
        color: #cbd5e1;
    }
    
    /* Professional Cards */
    .welcome-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 3rem;
        margin: 2rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .dark-mode .welcome-container {
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced Sidebar Styling for Better Visibility */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        color: #f8fafc !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #f8fafc !important;
    }
    
    .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2,
    .css-1d391kg .stMarkdown h3,
    .css-1d391kg .stMarkdown h4,
    .css-1d391kg .stMarkdown h5,
    .css-1d391kg .stMarkdown h6 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    .css-1d391kg .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    /* Enhanced Checkbox and Selectbox Visibility */
    .css-1d391kg .stCheckbox > label {
        color: #f8fafc !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox > label {
        color: #f8fafc !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(45deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* File Uploader Enhancement */
    .css-1d391kg .stFileUploader > div {
        border: 2px solid #3b82f6 !important;
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.1);
    }
    
    .css-1d391kg .stFileUploader > div:hover {
        border-color: #1d4ed8 !important;
        background: rgba(59, 130, 246, 0.2);
    }
    
    /* Selectbox Enhancement */
    .css-1d391kg .stSelectbox > div > div {
        border: 2px solid #3b82f6 !important;
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.1);
    }
    
    .css-1d391kg .stSelectbox > div > div:hover {
        border-color: #1d4ed8 !important;
        background: rgba(59, 130, 246, 0.2);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    .dark-mode .feature-card {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .dark-mode .feature-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: block;
        text-align: center;
        color: #3b82f6;
    }
    
    .dark-mode .feature-icon {
        color: #60a5fa;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .dark-mode .feature-title {
        color: #f1f5f9;
    }
    
    .feature-description {
        color: #64748b;
        text-align: center;
        line-height: 1.6;
        font-weight: 400;
        font-size: 1rem;
    }
    
    .dark-mode .feature-description {
        color: #cbd5e1;
    }
    
    /* Professional Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: #3b82f6;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .dark-mode .metric-card {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .dark-mode .metric-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .dark-mode .metric-card h2 {
        color: #f1f5f9;
    }
    
    .metric-card h3 {
        font-size: 1.5rem;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    
    .dark-mode .metric-card h3 {
        color: #60a5fa;
    }
    
    .metric-card p {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        margin: 0;
    }
    
    .dark-mode .metric-card p {
        color: #cbd5e1;
    }
    
    /* Professional Step Cards */
    .step-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        color: #1e293b;
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        transform: translateX(4px);
        border-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .dark-mode .step-card {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-left: 4px solid #60a5fa;
        color: #f1f5f9;
    }
    
    .dark-mode .step-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .step-number {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.2rem;
        margin-right: 1rem;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    /* Professional Use Case Cards */
    .use-case-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        color: #1e293b;
    }
    
    .use-case-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .dark-mode .use-case-card {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #f1f5f9;
    }
    
    .dark-mode .use-case-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Professional Insight Boxes */
    .insight-box {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #1e293b;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .dark-mode .insight-box {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #f1f5f9;
    }
    
    .dark-mode .insight-box:hover {
        border-color: #60a5fa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Professional Advanced Sections */
    .advanced-section {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .advanced-section:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .dark-mode .advanced-section {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .dark-mode .advanced-section:hover {
        border-color: #60a5fa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Professional Badges */
    .badge {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        margin: 0.25rem;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Professional Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5, .dark-mode h6 {
        color: #f1f5f9 !important;
    }
    
    .stMarkdown p {
        color: #64748b;
        line-height: 1.6;
    }
    
    .dark-mode .stMarkdown p {
        color: #cbd5e1;
    }
    
    /* Professional Form Elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        color: #1e293b;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 8px;
        font-weight: 500;
    }
    
    .dark-mode .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.9);
        color: #f1f5f9;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    
    .stCheckbox > label {
        color: #1e293b !important;
        font-weight: 500;
    }
    
    .dark-mode .stCheckbox > label {
        color: #f1f5f9 !important;
    }
    
    .stFileUploader > div {
        border: 2px dashed rgba(148, 163, 184, 0.4);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.5);
    }
    
    .dark-mode .stFileUploader > div {
        border: 2px dashed rgba(148, 163, 184, 0.5);
        background: rgba(30, 41, 59, 0.5);
    }
    
    /* Professional Sidebar */
    .sidebar-header {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Professional Privacy Badge */
    .privacy-badge {
        background: linear-gradient(45deg, #10b981, #059669);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    /* Section Spacing */
    .section-spacer {
        margin: 1.5rem 0;
        padding: 1rem 0;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .dark-mode .section-spacer {
        border-top: 1px solid rgba(148, 163, 184, 0.3);
        border-bottom: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .section-gap {
        margin: 1rem 0;
        padding: 0.5rem 0;
    }
    
    .subsection-gap {
        margin: 0.75rem 0;
        padding: 0.25rem 0;
    }
    
    /* Enhanced section headers */
    .section-header {
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #3b82f6;
        position: relative;
    }
    
    .dark-mode .section-header {
        border-bottom: 2px solid #60a5fa;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 50px;
        height: 2px;
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
    }
    
    .dark-mode .section-header::after {
        background: linear-gradient(45deg, #60a5fa, #3b82f6);
    }
    
</style>
""", unsafe_allow_html=True)

# Main title with professional styling
st.markdown('<h1 class="main-header">WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform your conversations into actionable insights with advanced analytics and AI-powered features</p>', unsafe_allow_html=True)

# Theme toggle
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.markdown('<div class="sidebar-header">🎯 Your Analysis Dashboard</div>', unsafe_allow_html=True)
with col2:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Apply theme to the app
if st.session_state.dark_mode:
    st.markdown("""
    <script>
        document.querySelector('.stApp').classList.add('dark-mode');
    </script>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <script>
        document.querySelector('.stApp').classList.remove('dark-mode');
    </script>
    """, unsafe_allow_html=True)

# Function to get chart colors based on theme
def get_chart_colors():
    if st.session_state.dark_mode:
        return {
            'bg_color': 'rgba(30, 41, 59, 0.95)',
            'paper_bg_color': 'rgba(30, 41, 59, 0.95)',
            'font_color': '#f8fafc',
            'grid_color': 'rgba(148, 163, 184, 0.3)',
            'line_colors': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
            'title_color': '#f1f5f9',
            'axis_color': '#cbd5e1'
        }
    else:
        return {
            'bg_color': 'rgba(255, 255, 255, 1)',
            'paper_bg_color': 'rgba(255, 255, 255, 1)',
            'font_color': '#1e293b',
            'grid_color': 'rgba(148, 163, 184, 0.3)',
            'line_colors': ['#1e40af', '#059669', '#d97706', '#dc2626', '#7c3aed'],
            'title_color': '#0f172a',
            'axis_color': '#475569'
        }

uploaded_file = st.sidebar.file_uploader("📁 Upload Your Chat File", type=['txt'], help="Upload WhatsApp, Telegram, or any chat export file")

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # User selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("👤 Select User/Analysis Type", user_list)
    
    # Analysis options with better organization
    st.sidebar.markdown("### 🔍 Analysis Categories")
    
    # Select All button
    if st.sidebar.button("✅ Select All Features", type="secondary", help="Select all analysis features at once"):
        st.session_state.select_all = True
        st.rerun()
    
    # Clear All button
    if st.sidebar.button("❌ Clear All Features", type="secondary", help="Clear all analysis features"):
        st.session_state.select_all = False
        st.rerun()
    
    # Basic Analytics
    st.sidebar.markdown("#### 📊 Basic Analytics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        basic_stats = st.checkbox("📈 Statistics", value=True if not st.session_state.get('select_all', False) else True)
        timeline = st.checkbox("📅 Timeline", value=True if not st.session_state.get('select_all', False) else True)
        emoji_analysis = st.checkbox("😊 Emojis", value=True if not st.session_state.get('select_all', False) else True)
    with col2:
        word_analysis = st.checkbox("📝 Words", value=True if not st.session_state.get('select_all', False) else True)
        activity_analysis = st.checkbox("🔥 Activity", value=True if not st.session_state.get('select_all', False) else True)
        user_analysis = st.checkbox("👥 Users", value=True if not st.session_state.get('select_all', False) else True)
    
    # Advanced Analytics
    st.sidebar.markdown("#### 🧠 Advanced Analytics")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        sentiment_checkbox = st.checkbox("😊 Sentiment", value=True if st.session_state.get('select_all', False) else False)
        response_analysis = st.checkbox("⚡ Response", value=True if st.session_state.get('select_all', False) else False)
        style_analysis = st.checkbox("💬 Style", value=True if st.session_state.get('select_all', False) else False)
        topics_analysis = st.checkbox("🎯 Topics", value=True if st.session_state.get('select_all', False) else False)
    with col4:
        dynamics_analysis = st.checkbox("🏆 Dynamics", value=True if st.session_state.get('select_all', False) else False)
        insights_analysis = st.checkbox("🧠 AI Insights", value=True if st.session_state.get('select_all', False) else False)
        predictions_analysis = st.checkbox("🔮 Predictions", value=True if st.session_state.get('select_all', False) else False)
        report_analysis = st.checkbox("📋 Report", value=True if st.session_state.get('select_all', False) else False)
    
    # Privacy & Security
    st.sidebar.markdown("#### 🔒 Privacy & Security")
    privacy_mode = st.sidebar.checkbox("🕵️ Anonymous Mode", value=False, help="Analyze patterns without showing actual messages")

    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        
        # Show loading message
        with st.spinner("🔄 Processing your chat data... This may take a few moments for large files."):
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Apply privacy mode if enabled
            if privacy_mode:
                status_text.text("🔒 Enabling privacy mode...")
                progress_bar.progress(10)
                st.info("🔒 Privacy Mode Enabled: All analysis will be performed anonymously without showing actual message content.")
                df_analysis, user_mapping = create_anonymous_analysis(df)
                progress_bar.progress(30)
                # Show user mapping for reference
                with st.expander("👥 User Mapping (for reference)"):
                    for original, anonymous in user_mapping.items():
                        if original != 'group_notification':
                            st.write(f"{original} → {anonymous}")
            else:
                status_text.text("📊 Loading data...")
                progress_bar.progress(20)
                df_analysis = df
        
        # Basic Statistics
        if basic_stats:
            st.header("📊 Chat Statistics Overview")
            
            num_messages, words, num_media_messages, num_links = get_cached_stats(selected_user, df_analysis)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💬</h3>
                    <h2>{num_messages:,}</h2>
                    <p>Messages</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝</h3>
                    <h2>{words:,}</h2>
                    <p>Words</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📷</h3>
                    <h2>{num_media_messages:,}</h2>
                    <p>Media</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔗</h3>
                    <h2>{num_links:,}</h2>
                    <p>Links</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

        # Timeline Analysis
        if timeline:
            st.header("📈 Timeline Analysis")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 Monthly Timeline")
                timeline_data = monthly_timeline(selected_user, df_analysis)
                colors = get_chart_colors()
                fig = px.line(timeline_data, x='time', y='message', 
                             title='Messages Over Time (Monthly)',
                             labels={'message': 'Message Count', 'time': 'Month-Year'})
                fig.update_traces(line_color=colors['line_colors'][0], line_width=3)
                fig.update_layout(
                    plot_bgcolor=colors['bg_color'],
                    paper_bgcolor=colors['paper_bg_color'],
                    font=dict(color=colors['font_color'], size=12),
                    xaxis=dict(
                        gridcolor=colors['grid_color'],
                        tickfont=dict(color=colors['axis_color']),
                        title=dict(font=dict(color=colors['axis_color']))
                    ),
                    yaxis=dict(
                        gridcolor=colors['grid_color'],
                        tickfont=dict(color=colors['axis_color']),
                        title=dict(font=dict(color=colors['axis_color']))
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Daily Timeline")
                daily_timeline_data = daily_timeline(selected_user, df_analysis)
                fig = px.line(daily_timeline_data, x='only_date', y='message',
                             title='Messages Over Time (Daily)',
                             labels={'message': 'Message Count', 'only_date': 'Date'})
                fig.update_traces(line_color=colors['line_colors'][1], line_width=3)
                fig.update_layout(
                    plot_bgcolor=colors['bg_color'],
                    paper_bgcolor=colors['paper_bg_color'],
                    font=dict(color=colors['font_color'], size=12),
                    xaxis=dict(
                        gridcolor=colors['grid_color'],
                        tickfont=dict(color=colors['axis_color']),
                        title=dict(font=dict(color=colors['axis_color']))
                    ),
                    yaxis=dict(
                        gridcolor=colors['grid_color'],
                        tickfont=dict(color=colors['axis_color']),
                        title=dict(font=dict(color=colors['axis_color']))
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

        # Emoji Analysis
        if emoji_analysis:
            st.header("😊 Emoji Analysis")
            try:
                emoji_df = emoji_helper(selected_user, df_analysis)
                if not emoji_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("😊 Top Emojis Used")
                        st.dataframe(emoji_df.head(10), use_container_width=True)
                    with col2:
                        fig = px.pie(emoji_df.head(8), values='count', names='emoji',
                                   title='Top Emojis Used',
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No emojis found in the selected data.")
            except:
                st.warning("Could not analyze emojis.")
            
            st.markdown("---")

        # Word Analysis
        if word_analysis:
            st.header("📝 Word Analysis")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Most Common Words")
                try:
                    most_common_df = most_common_words(selected_user, df_analysis)
                    if not most_common_df.empty:
                        colors = get_chart_colors()
                        fig = px.bar(most_common_df, x=1, y=0, orientation='h',
                                   title='Top 20 Most Common Words',
                                   color=1,
                                   color_continuous_scale='Viridis')
                        fig.update_layout(
                            showlegend=False,
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Could not analyze common words.")
            
            with col2:
                st.subheader("☁️ Word Cloud")
                try:
                    df_wc = create_wordcloud(selected_user, df_analysis)
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_facecolor('none')
                    fig.patch.set_alpha(0)
                    st.pyplot(fig)
                except:
                    st.warning("Could not generate word cloud.")
            
            st.markdown("---")

        # Activity Analysis
        if activity_analysis:
            st.header("🔥 Activity Analysis")
            
            # Most Busy Users (for group chats)
            if selected_user == 'Overall' and len(df_analysis['user'].unique()) > 2:
                st.subheader("👥 Most Active Users")
                x, new_df = most_busy_users(df_analysis)
                
                col1, col2 = st.columns(2)
                with col1:
                    colors = get_chart_colors()
                    fig = px.bar(x=x.values, y=x.index, orientation='h',
                               title='Most Active Users (Message Count)',
                               labels={'x': 'Message Count', 'y': 'User'},
                               color=x.values,
                               color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['paper_bg_color'],
                        font=dict(color=colors['font_color'], size=12),
                        xaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        ),
                        yaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Activity Percentage")
                    st.dataframe(new_df, use_container_width=True)
            
            st.markdown("---")

        # User Analysis
        if user_analysis:
            st.header("👥 User Analysis")
            
            if selected_user == 'Overall' and len(df_analysis['user'].unique()) > 2:
                st.subheader("📊 User Statistics")
                user_stats = df_analysis[df_analysis['user'] != 'group_notification']['user'].value_counts()
                
                colors = get_chart_colors()
                fig = px.pie(values=user_stats.values, names=user_stats.index,
                           title='Message Distribution by User',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(
                    plot_bgcolor=colors['bg_color'],
                    paper_bgcolor=colors['paper_bg_color'],
                    font=dict(color=colors['font_color'], size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

        # ==================== ADVANCED ANALYTICS ====================

        # Enhanced section divider for major sections
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        # Sentiment Analysis
        if sentiment_checkbox:
            st.header("😊 Sentiment & Emotion Analysis")
            
            try:
                sentiment_data = sentiment_analysis(selected_user, df_analysis)
                
                if not sentiment_data.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Sentiment distribution
                        sentiment_counts = sentiment_data['sentiment'].value_counts()
                        colors = get_chart_colors()
                        fig = px.pie(values=sentiment_counts.values, 
                                   names=sentiment_counts.index,
                                   title='Overall Sentiment Distribution',
                                   color_discrete_map={
                                       'Positive': '#10b981',
                                       'Negative': '#ef4444',
                                       'Neutral': '#f59e0b'
                                   })
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Sentiment metrics
                        avg_polarity = sentiment_data['polarity'].mean()
                        avg_subjectivity = sentiment_data['subjectivity'].mean()
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊</h3>
                            <h2>{avg_polarity:.3f}</h2>
                            <p>Avg Polarity</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎭</h3>
                            <h2>{avg_subjectivity:.3f}</h2>
                            <p>Avg Subjectivity</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sentiment interpretation
                        if avg_polarity > 0.1:
                            st.success("😊 Overall Positive Communication")
                        elif avg_polarity < -0.1:
                            st.error("😔 Overall Negative Communication")
                        else:
                            st.info("😐 Neutral Communication Tone")
                    
                    # Sentiment timeline
                    st.subheader("📈 Sentiment Over Time")
                    sentiment_timeline_data = sentiment_timeline(selected_user, df_analysis)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sentiment_timeline_data['only_date'], 
                                           y=sentiment_timeline_data['positive_ratio'],
                                           mode='lines', name='Positive Ratio',
                                           line=dict(color=colors['line_colors'][1], width=3)))
                    fig.add_trace(go.Scatter(x=sentiment_timeline_data['only_date'], 
                                           y=sentiment_timeline_data['negative_ratio'],
                                           mode='lines', name='Negative Ratio',
                                           line=dict(color=colors['line_colors'][0], width=3)))
                    fig.update_layout(
                        title=dict(
                            text='Sentiment Trends Over Time',
                            font=dict(color=colors['title_color'], size=16)
                        ),
                        xaxis_title='Date', 
                        yaxis_title='Ratio',
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['paper_bg_color'],
                        font=dict(color=colors['font_color'], size=12),
                        xaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        ),
                        yaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("Unable to perform sentiment analysis on this data.")
                    
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
            
            st.markdown("---")

        # Response Time Analysis
        if response_analysis:
            st.header("⚡ Response Time & Conversation Dynamics")
            
            try:
                response_df, avg_response = response_time_analysis(selected_user, df_analysis)
                
                if not response_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Response Time Statistics")
                        avg_resp_time = response_df['response_time_minutes'].mean()
                        median_resp_time = response_df['response_time_minutes'].median()
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏱️</h3>
                            <h2>{avg_resp_time:.1f}m</h2>
                            <p>Avg Response</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊</h3>
                            <h2>{median_resp_time:.1f}m</h2>
                            <p>Median Response</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Response time distribution
                        colors = get_chart_colors()
                        fig = px.histogram(response_df, x='response_time_minutes',
                                         title='Response Time Distribution',
                                         nbins=30, color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("👥 Response Time by User")
                        if not avg_response.empty:
                            fig = px.bar(x=avg_response.index, y=avg_response['mean'],
                                       title='Average Response Time by User',
                                       labels={'y': 'Avg Response Time (min)', 'x': 'User'},
                                       color=avg_response['mean'],
                                       color_continuous_scale='Viridis')
                            fig.update_layout(
                                plot_bgcolor=colors['bg_color'],
                                paper_bgcolor=colors['paper_bg_color'],
                                font=dict(color=colors['font_color'], size=12),
                                xaxis=dict(
                                    gridcolor=colors['grid_color'],
                                    tickfont=dict(color=colors['axis_color']),
                                    title=dict(font=dict(color=colors['axis_color']))
                                ),
                                yaxis=dict(
                                    gridcolor=colors['grid_color'],
                                    tickfont=dict(color=colors['axis_color']),
                                    title=dict(font=dict(color=colors['axis_color']))
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Conversation initiator analysis
                st.subheader("🎯 Conversation Initiators")
                initiator_data = conversation_initiator_analysis(selected_user, df_analysis)
                
                if not initiator_data.empty:
                    fig = px.pie(values=initiator_data.values, names=initiator_data.index,
                               title='Who Starts Conversations Most Often?',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['paper_bg_color'],
                        font=dict(color=colors['font_color'], size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in response analysis: {str(e)}")
            
            st.markdown("---")

        # Communication Style Analysis
        if style_analysis:
            st.header("💬 Communication Style & Personality")
            
            try:
                # Message length analysis
                msg_data, length_stats = message_length_analysis(selected_user, df_analysis)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📏 Message Length Patterns")
                    if not length_stats.empty:
                        st.dataframe(length_stats, use_container_width=True)
                    
                    # Message length distribution
                    if not msg_data.empty:
                        colors = get_chart_colors()
                        fig = px.box(msg_data, x='user', y='message_length',
                                   title='Message Length Distribution by User',
                                   color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎨 Communication Style Metrics")
                    style_data = communication_style_analysis(selected_user, df_analysis)
                    
                    if not style_data.empty:
                        st.dataframe(style_data, use_container_width=True)
                        
                        # Create a simpler visualization instead of radar
                        fig = px.bar(style_data.reset_index(), 
                                   x='user', y='exclamation_count',
                                   title='Exclamation Usage by User',
                                   color='exclamation_count',
                                   color_continuous_scale='Viridis')
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in communication style analysis: {str(e)}")
            
            st.markdown("---")

        # Topic Analysis
        if topics_analysis:
            st.header("🎯 Smart Content & Topic Analysis")
            
            try:
                topics_data, lda_model = topic_modeling(selected_user, df_analysis)
                
                if topics_data:
                    st.subheader("🔍 Discovered Topics")
                    for i, topic in enumerate(topics_data):
                        st.write(f"**Topic {i+1}:** {', '.join(topic['words'][:5])}")
                    
                    # Topic visualization
                    topic_df = pd.DataFrame(topics_data)
                    colors = get_chart_colors()
                    fig = px.bar(topic_df, x='topic_id', y='weight',
                               title='Topic Strength Distribution',
                               color='weight',
                               color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['paper_bg_color'],
                        font=dict(color=colors['font_color'], size=12),
                        xaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        ),
                        yaxis=dict(
                            gridcolor=colors['grid_color'],
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for topic modeling or analysis failed.")
                
                # Important moments detection
                st.subheader("⭐ Important Moments")
                important_moments = detect_important_moments(selected_user, df_analysis)
                
                if not important_moments.empty:
                    st.dataframe(important_moments.head(10)[['date', 'user', 'message', 'score']], use_container_width=True)
                else:
                    st.info("No significant moments detected.")
                
            except Exception as e:
                st.error(f"Error in topic analysis: {str(e)}")
            
            st.markdown("---")

        # Group Dynamics (only for group chats)
        if dynamics_analysis and len(df_analysis['user'].unique()) > 3:
            st.header("🏆 Group Dynamics & Social Network")
            
            try:
                # Interaction Matrix
                st.subheader("🤝 Interaction Matrix")
                interaction_matrix = group_interaction_matrix(df_analysis)
                
                if not interaction_matrix.empty:
                    colors = get_chart_colors()
                    fig = px.imshow(interaction_matrix,
                                  title='Who Responds to Whom?',
                                  labels=dict(color="Interactions"),
                                  color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['paper_bg_color'],
                        font=dict(color=colors['font_color'], size=12),
                        xaxis=dict(
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        ),
                        yaxis=dict(
                            tickfont=dict(color=colors['axis_color']),
                            title=dict(font=dict(color=colors['axis_color']))
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Group roles
                st.subheader("👥 Group Roles")
                roles = identify_group_roles(df_analysis)
                
                role_summary = []
                for user, role_data in roles.items():
                    role_summary.append({
                        'User': user,
                        'Primary Role': role_data['primary_role'],
                        'Messages': role_data['stats']['total_messages'],
                        'Response Rate': f"{role_data['stats']['response_rate']:.2%}"
                    })
                
                roles_df = pd.DataFrame(role_summary)
                st.dataframe(roles_df, use_container_width=True)
                
                # Role distribution
                role_counts = roles_df['Primary Role'].value_counts()
                colors = get_chart_colors()
                fig = px.pie(values=role_counts.values, names=role_counts.index,
                           title='Group Role Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(
                    plot_bgcolor=colors['bg_color'],
                    paper_bgcolor=colors['paper_bg_color'],
                    font=dict(color=colors['font_color'], size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in group dynamics analysis: {str(e)}")
            
            st.markdown("---")

        # AI Insights
        if insights_analysis:
            st.header("🧠 AI-Generated Insights")
            
            try:
                insights_data = generate_chat_insights(selected_user, df_analysis)
                
                for insight in insights_data:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                
                # Conversation highlights
                st.subheader("💎 Conversation Highlights")
                highlights = conversation_highlights(selected_user, df_analysis)
                
                if not highlights.empty:
                    for _, highlight in highlights.head(5).iterrows():
                        with st.expander(f"📅 {highlight['date'].strftime('%Y-%m-%d %H:%M')} - {highlight['user']}"):
                            st.write(highlight['message'])
                            st.caption(f"Importance Score: {highlight['score']}")
                
                # Predictions
                st.subheader("🔮 Activity Predictions")
                predictions = predict_activity_patterns(df_analysis)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Peak Hours:**")
                    for hour in predictions['peak_hours']:
                        st.write(f"• {hour}:00")
                
                with col2:
                    st.write("**Quiet Hours:**")
                    for hour in predictions['quiet_hours']:
                        st.write(f"• {hour}:00")
                
                # Relationship evolution
                if selected_user != 'Overall':
                    st.subheader("📈 Relationship Evolution")
                    evolution = relationship_evolution_analysis(selected_user, df_analysis)
                    
                    if not evolution.empty:
                        colors = get_chart_colors()
                        fig = px.line(evolution, x='period', y=['message_count', 'avg_sentiment'],
                                    title='Communication Evolution Over Time')
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in AI insights: {str(e)}")
            
            st.markdown("---")

        # Gamification & Badges
        if len(df_analysis['user'].unique()) > 2:
            st.header("🎮 Gamification & Achievements")
            
            try:
                badges = calculate_communication_badges(selected_user, df_analysis)
                
                # Filter users who have badges
                users_with_badges = {user: user_badges for user, user_badges in badges.items() if len(user_badges) > 0}
                
                if users_with_badges:
                    st.subheader("🏆 User Badges")
                    for user, user_badges in users_with_badges.items():
                        with st.expander(f"🎯 {user}'s Achievements ({len(user_badges)} badges)"):
                            for badge in user_badges:
                                st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
                else:
                    st.info("No badges earned yet. Keep chatting to unlock achievements!")
                
                # Personality matching
                if len(df_analysis['user'].unique()) > 2:
                    st.subheader("🧬 Personality Matching")
                    similarities, personalities = personality_matching_analysis(df_analysis)
                    
                    if not similarities.empty:
                        st.dataframe(similarities.head(), use_container_width=True)
                        
                        # Compatibility chart
                        colors = get_chart_colors()
                        fig = px.bar(similarities.head(10), x='similarity_score', 
                                   y=[f"{row['user1']} - {row['user2']}" for _, row in similarities.head(10).iterrows()],
                                   orientation='h',
                                   title='Top User Compatibility Matches',
                                   color='similarity_score',
                                   color_continuous_scale='Viridis')
                        fig.update_layout(
                            plot_bgcolor=colors['bg_color'],
                            paper_bgcolor=colors['paper_bg_color'],
                            font=dict(color=colors['font_color'], size=12),
                            xaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            ),
                            yaxis=dict(
                                gridcolor=colors['grid_color'],
                                tickfont=dict(color=colors['axis_color']),
                                title=dict(font=dict(color=colors['axis_color']))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in gamification analysis: {str(e)}")
            
            st.markdown("---")

        # Enhanced section divider for report section
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        # Comprehensive Report
        if report_analysis:
            st.header("📋 Comprehensive Analysis Report")
            
            try:
                report_data = generate_comprehensive_report(selected_user, df_analysis)
                
                # Display report sections
                if 'basic_stats' in report_data:
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>💬</h3>
                            <h2>{report_data['basic_stats']['total_messages']:,}</h2>
                            <p>Messages</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📝</h3>
                            <h2>{report_data['basic_stats']['total_words']:,}</h2>
                            <p>Words</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📷</h3>
                            <h2>{report_data['basic_stats']['media_messages']:,}</h2>
                            <p>Media</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🔗</h3>
                            <h2>{report_data['basic_stats']['links_shared']:,}</h2>
                            <p>Links</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if 'sentiment_analysis' in report_data and report_data['sentiment_analysis']:
                    st.subheader("😊 Sentiment Summary")
                    sentiment = report_data['sentiment_analysis']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>😊</h3>
                            <h2>{sentiment['positive_ratio']:.1%}</h2>
                            <p>Positive</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>😐</h3>
                            <h2>{sentiment['neutral_ratio']:.1%}</h2>
                            <p>Neutral</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>😔</h3>
                            <h2>{sentiment['negative_ratio']:.1%}</h2>
                            <p>Negative</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if 'insights' in report_data:
                    st.subheader("🔍 Key Insights")
                    for insight in report_data['insights']:
                        st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)
                
                # Download report as JSON
                st.subheader("💾 Export Report")
                
                # Create the report JSON
                import json
                report_json = json.dumps(report_data, indent=2, default=str)
                
                # Create CSV report for basic stats
                csv_data = ""
                if 'basic_stats' in report_data:
                    csv_data += "Metric,Value\n"
                    csv_data += f"Total Messages,{report_data['basic_stats']['total_messages']}\n"
                    csv_data += f"Total Words,{report_data['basic_stats']['total_words']}\n"
                    csv_data += f"Media Messages,{report_data['basic_stats']['media_messages']}\n"
                    csv_data += f"Links Shared,{report_data['basic_stats']['links_shared']}\n"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON Download button
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=report_json,
                        file_name=f"chat_analysis_report_{selected_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download a comprehensive JSON report of all analysis results"
                    )
                
                with col2:
                    # CSV Download button
                    if csv_data:
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv_data,
                            file_name=f"chat_analysis_summary_{selected_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download a simple CSV summary of key metrics"
                        )
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

else:
    # Simple, clean welcome message without repeated content
    st.markdown("""
    <div class="welcome-container">
        <h2 style="color: #1e293b; margin-bottom: 2rem;">🚀 Ready to Analyze Your Chat?</h2>
        <p style="font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 3rem;">
            Upload your WhatsApp chat export to get started with comprehensive analytics and insights.
        </p>
        <div style="text-align: center;">
            <p style="color: #64748b; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> Export your chat from WhatsApp → More options → Export chat
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)