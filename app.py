import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import plotly.express as px

# Set page configuration for a professional look
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a clean, professional dark theme with responsive design ---
st.markdown("""
<style>
    /* Main container background */
    .reportview-container {
        background: #121820; /* Dark Blue */
    }
    
    /* Sidebar background */
    .sidebar .sidebar-content {
        background: #1e2a3a; /* Slightly lighter dark blue */
        color: #F0F8FF; /* White */
    }
    
    /* Title and Header colors */
    h1, h2, h3, h4, h5, h6, .st-b5, .st-b6 {
        color: #00FFFF; /* Vibrant Cyan */
    }
    
    /* Text color for the main content */
    body {
        color: #F0F8FF; 
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #00FFFF; /* Cyan */
        color: #121820;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #00CCCC; /* Darker Cyan */
    }
    
    /* Selectbox and File Uploader styling */
    .stSelectbox, .stFileUploader {
        background-color: #2b3a4a; /* Darker grey-blue for inputs */
        border: 1px solid #4a647d;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: #F0F8FF;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2b3a4a;
        color: #F0F8FF;
    }
    .stSelectbox div[role="listbox"] {
        background-color: #2b3a4a;
        color: #F0F8FF;
    }

    /* Metric boxes styling */
    .stMetric, .stAlert {
        background-color: #2b3a4a;
        color: #F0F8FF;
        border-radius: 10px;
        border: 1px solid #4a647d;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        padding: 20px;
        margin-bottom: 15px;
    }
    div[data-testid="stMetricLabel"] {
        color: #90CAF9 !important; /* Light blue for labels */
        font-size: 0.9em;
    }
    div[data-testid="stMetricValue"] {
        color: #FF8C00 !important; /* Vibrant Orange for values */
        font-size: 1.8em;
        font-weight: bold;
    }

    /* Dataframe styling */
    .stDataFrame {
        background-color: #2b3a4a;
        color: #F0F8FF;
        border-radius: 10px;
        border: 1px solid #4a647d;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* Info and Warning boxes */
    div.stAlert.st-bc {
        background-color: #1e2a3a;
        border-left: 5px solid #00FFFF;
    }
    div.stAlert.st-bd {
        background-color: #1e2a3a;
        border-left: 5px solid #FF8C00;
    }
    
    /* === Responsive Design: Media Queries === */
    @media (max-width: 768px) {
        /* Styles for tablets and mobile devices */
        h1 {
            font-size: 2em; /* Slightly smaller main title */
        }
        h2 {
            font-size: 1.5em; /* Smaller sub-headers */
        }
        .stMetric > div[data-testid="stMetricLabel"] {
            font-size: 0.8em; /* Smaller metric labels */
        }
        .stMetric > div[data-testid="stMetricValue"] {
            font-size: 1.5em !important; /* Smaller metric values */
        }
        .stMetric, .stAlert {
            padding: 15px; /* Less padding for compactness */
        }
        .stButton>button {
            padding: 8px 16px; /* Smaller button padding */
        }
    }
    
    @media (max-width: 480px) {
        /* Styles for mobile phones */
        h1 {
            font-size: 1.5em; /* Even smaller main title */
        }
        h2 {
            font-size: 1.2em; /* Smaller sub-headers */
        }
        .stMetric > div[data-testid="stMetricValue"] {
            font-size: 1.2em !important; /* Smallest metric values */
        }
        .stMetric, .stAlert {
            padding: 10px; /* Minimal padding */
        }
    }
</style>
""", unsafe_allow_html=True)

# Main title for the landing page
st.title("WhatsApp Chat Analyzer 💬")
st.markdown("Transform your conversations into actionable insights with advanced analytics and AI-powered features.")
st.markdown("---")

# Sidebar for file upload and controls
st.sidebar.title("💬 WhatsApp Chat Analyzer")
st.sidebar.markdown("---")
st.sidebar.header("Upload your exported file")
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Conditional logic to show either the landing page or the dashboard
if uploaded_file is None:
    # --- Landing Page UI ---
    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # --- Instructions Section ---
    st.header("How to Use This Analyzer")
    st.markdown("Follow these simple steps to analyze your WhatsApp chat.")
    
    st.info("### 1. Export Your Chat from WhatsApp")
    st.markdown("On your phone, open the individual or group chat you wish to analyze and follow these steps:")
    st.markdown("1. Go to (three dots) or **Group info**.")
    st.markdown("2. Select **More**.")
    st.markdown("3. Tap **Export chat**.")
    st.markdown("4. Choose **Without media** to ensure the file size is small and the analysis is fast.")
    st.markdown("5. Share the `.txt` file to your email, cloud storage, or directly to your computer.")
    
    st.warning("### 2. Upload the File")
    st.markdown("Use the **'Browse files'** button in the sidebar on the left to upload the `.txt` file you just exported.")
    
    st.success("### 3. Generate Analysis")
    st.markdown("Once the file is uploaded, a dropdown menu will appear in the sidebar. You can select a specific user or choose 'Overall' to see group-level analysis. Finally, click the **'Show Analysis'** button to generate the dashboard.")

    # --- What You Can Analyze Section ---
    st.markdown("---")
    st.header("What You Can Analyze from Your Chat")
    
    st.markdown("""
    This analyzer provides deep insights into your conversations, including:

    * **Top Statistics**: Get a quick overview of total messages, words, media shared, and links sent.
    * **Activity Timelines**: See your chat activity visualized by month and day to understand when conversations are most active.
    * **User Activity**: Identify the most active users in a group chat and view their communication share.
    * **Word & Emoji Insights**: Discover the most common words and emojis used, and see a word cloud of your chat.
    * **Sentiment Analysis**: Understand the emotional tone of your conversations (positive, negative, or neutral).
    * **Response Time Analysis**: See who responds fastest and analyze communication flow.
    * **Communication Style**: Analyze average message length, word count, and other stylistic features.
    * **Topic Modeling**: Automatically identify the main subjects and themes discussed in the chat.
    """)

else:
    # --- Analysis Options in Sidebar ---
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for:", user_list)

    # New Checkbox features 
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Categories")
    
    # Checkbox for selecting all features
    select_all = st.sidebar.checkbox("Select All Features", value=True)
    if st.sidebar.button("Clear All Features"):
        select_all = False
    
    st.sidebar.subheader("Basic Analytics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        basic_stats = st.checkbox("Statistics", value=select_all, key="stats")
        timeline = st.checkbox("Timeline", value=select_all, key="timeline")
        emojis = st.checkbox("Emojis", value=select_all, key="emojis")
    with col2:
        words = st.checkbox("Words", value=select_all, key="words")
        activity = st.checkbox("Activity", value=select_all, key="activity")
        users = st.checkbox("Users", value=select_all, key="users")
    
    st.sidebar.subheader("Advanced Analytics")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        sentiment = st.checkbox("Sentiment", value=False, key="sentiment")
        response = st.checkbox("Response", value=False, key="response")
        style = st.checkbox("Style", value=False, key="style")
        topics = st.checkbox("Topics", value=False, key="topics")
    
    # Placeholder for the new and upcoming features
    st.sidebar.subheader("Upcoming Features")
    st.sidebar.markdown("`Dynamics` `AI Insights` `Predictions` `Report` `Anonymous Mode`")

    if st.sidebar.button("Run Analysis"):
        st.title("📊 Chat Analysis Dashboard")

        # Top Statistics with colorful icons
        if basic_stats:
            st.header("✨ Top Statistics")
            num_messages, total_words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Total Messages", value=f"{num_messages:,}")
            with col2:
                st.metric(label="Total Words", value=f"{total_words:,}")
            with col3:
                st.metric(label="Media Shared", value=f"{num_media_messages:,}")
            with col4:
                st.metric(label="Links Shared", value=f"{num_links:,}")

        # Monthly Timeline
        if timeline:
            st.header("📈 Monthly Activity")
            timeline_df = helper.monthly_timeline(selected_user, df)
            
            fig = px.line(timeline_df, 
                          x='time', 
                          y='message', 
                          title='Monthly Activity Timeline',
                          labels={'message': 'Number of Messages', 'time': 'Month-Year'},
                          template='plotly_dark')
            
            fig.update_layout(
                xaxis_title_font_color='white',
                yaxis_title_font_color='white',
                title_font_color='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            # Daily Timeline
            st.header("📅 Daily Activity")
            daily_timeline_df = helper.daily_timeline(selected_user, df)
            
            fig = px.line(daily_timeline_df, 
                          x='only_date', 
                          y='message', 
                          title='Daily Activity Timeline',
                          labels={'message': 'Number of Messages', 'only_date': 'Date'},
                          template='plotly_dark')

            fig.update_layout(
                xaxis_title_font_color='white',
                yaxis_title_font_color='white',
                title_font_color='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity Map
        if activity:
            st.header("🗓️ Activity Maps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Busy Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig = px.bar(busy_day,
                             x=busy_day.index,
                             y=busy_day.values,
                             title='Most Busy Day',
                             labels={'x': 'Day of Week', 'y': 'Number of Messages'},
                             template='plotly_dark',
                             color_discrete_sequence=['#FF4500'])
                fig.update_layout(
                    xaxis_title_font_color='white',
                    yaxis_title_font_color='white',
                    title_font_color='white',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("Most Busy Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig = px.bar(busy_month,
                             x=busy_month.index,
                             y=busy_month.values,
                             title='Most Busy Month',
                             labels={'x': 'Month', 'y': 'Number of Messages'},
                             template='plotly_dark',
                             color_discrete_sequence=['#8A2BE2'])
                fig.update_layout(
                    xaxis_title_font_color='white',
                    yaxis_title_font_color='white',
                    title_font_color='white',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Weekly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig = px.imshow(user_heatmap,
                            title="Weekly Activity Heatmap",
                            labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
                            color_continuous_scale='YlGnBu')
            fig.update_layout(
                xaxis_title_font_color='white',
                yaxis_title_font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Most busy users (Group level)
        if users:
            if selected_user == 'Overall':
                st.header("👥 Most Busy Users")
                x, new_df = helper.most_busy_users(df)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x,
                                 x=x.index,
                                 y=x.values,
                                 title='Most Busy Users',
                                 labels={'x': 'User', 'y': 'Number of Messages'},
                                 template='plotly_dark',
                                 color_discrete_sequence=['#00CED1'])
                    fig.update_layout(
                        xaxis_title_font_color='white',
                        yaxis_title_font_color='white',
                        title_font_color='white',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(new_df)

        # WordCloud and Most Common Words
        if words:
            st.header("☁️ Word Cloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(df_wc)
            ax.axis("off")
            fig.patch.set_facecolor('#121820')
            st.pyplot(fig)

            st.header("📝 Most Common Words")
            most_common_df = helper.most_common_words(selected_user, df)
            
            fig = px.bar(most_common_df,
                         x='Count', # Corrected from x=most_common_df[1]
                         y='Word',  # Corrected from y=most_common_df[0]
                         orientation='h',
                         title='Most Common Words',
                         labels={'Count': 'Count', 'Word': 'Words'},
                         template='plotly_dark',
                         color_discrete_sequence=['#FF6347'])
            fig.update_layout(
                xaxis_title_font_color='white',
                yaxis_title_font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Emoji analysis
        if emojis:
            emoji_df = helper.emoji_helper(selected_user, df)
            st.header("😂 Emoji Analysis")

            if not emoji_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(emoji_df)
                
                with col2:
                    fig = px.pie(emoji_df.head(5),
                                 names='emoji',
                                 values='count',
                                 title='Top 5 Emojis Used',
                                 template='plotly_dark',
                                 hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig)
            else:
                st.info("No emojis found for this user/chat.")

        st.markdown("---")
        st.header("🧠 Advanced Chat Insights")

        # 1. Sentiment Analysis
        if sentiment:
            st.subheader("Sentiment Analysis 😃🙁")
            sentiment_df = helper.sentiment_analysis(selected_user, df)
            sentiment_counts = sentiment_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            if not sentiment_counts.empty:
                fig = px.pie(sentiment_counts, 
                             names='Sentiment', 
                             values='Count', 
                             title='Sentiment Distribution',
                             template='plotly_dark',
                             hole=0.4,
                             color='Sentiment',
                             color_discrete_map={
                                'Positive': '#7FFF7F',  
                                'Neutral': '#6495ED',   
                                'Negative': '#FF6347'
                            })
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)
            else:
                st.info("Not enough message data for sentiment analysis.")

        # 2. Response Time Analysis
        if response:
            st.subheader("Response Time Analysis ⏱️")
            response_df, avg_response_by_user = helper.response_time_analysis(selected_user, df)
            if not response_df.empty:
                st.write("Average response time (in minutes):")
                st.dataframe(avg_response_by_user.round(2))
            else:
                st.info("Not enough data to analyze response times.")

        # 3. Message Length & Style Analysis
        if style:
            st.subheader("Communication Style ✍️")
            temp_df, length_stats = helper.message_length_analysis(selected_user, df)
            if not temp_df.empty:
                st.write("Average message length and word count:")
                st.dataframe(length_stats)

        # 4. Topic Modeling
        if topics:
            st.subheader("Top Conversation Topics 💬")
            topics_list, lda_model = helper.topic_modeling(selected_user, df)
            if topics_list:
                for topic in topics_list:
                    st.write(f"**Topic {topic['topic_id']}:** {', '.join(topic['words'])}")
            else:
                st.info("Not enough messages to perform topic modeling (min 10 messages).")