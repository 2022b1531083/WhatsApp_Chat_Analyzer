from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from collections import Counter
import emoji
import re
from datetime import datetime, timedelta
from textblob import TextBlob
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

extract = URLExtract()

# ==================== EXISTING FUNCTIONS ====================
def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):
    try:
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()
        f.close()
    except FileNotFoundError:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    try:
        f = open('stop_hinglish.txt','r')
        stop_words = f.read()
        f.close()
    except FileNotFoundError:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_counter = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counter.most_common(), columns=['emoji', 'count'])
    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

# ==================== NEW ADVANCED FEATURES ====================

# 1. SENTIMENT ANALYSIS FUNCTIONS
def sentiment_analysis(selected_user, df):
    """Analyze sentiment patterns in conversations"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n'].copy()
    
    sentiments = []
    polarities = []
    subjectivities = []
    
    for message in temp['message']:
        try:
            blob = TextBlob(message)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            polarities.append(polarity)
            subjectivities.append(subjectivity)
            
            if polarity > 0.1:
                sentiments.append('Positive')
            elif polarity < -0.1:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        except Exception:
            sentiments.append('Neutral')
            polarities.append(0)
            subjectivities.append(0)
    
    temp['sentiment'] = sentiments
    temp['polarity'] = polarities
    temp['subjectivity'] = subjectivities
    
    return temp

def sentiment_timeline(selected_user, df):
    """Create sentiment timeline"""
    sentiment_df = sentiment_analysis(selected_user, df)
    timeline = sentiment_df.groupby(['only_date', 'sentiment']).size().unstack(fill_value=0)
    timeline['total'] = timeline.sum(axis=1)
    timeline['positive_ratio'] = timeline.get('Positive', 0) / timeline['total']
    timeline['negative_ratio'] = timeline.get('Negative', 0) / timeline['total']
    return timeline.reset_index()

def emotion_patterns(selected_user, df):
    """Analyze emotional patterns"""
    sentiment_df = sentiment_analysis(selected_user, df)
    
    emotion_by_hour = sentiment_df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
    emotion_by_day = sentiment_df.groupby(['day_name', 'sentiment']).size().unstack(fill_value=0)
    
    avg_polarity = sentiment_df.groupby('user')['polarity'].mean().sort_values(ascending=False)
    avg_subjectivity = sentiment_df.groupby('user')['subjectivity'].mean().sort_values(ascending=False)
    
    return {
        'by_hour': emotion_by_hour,
        'by_day': emotion_by_day,
        'avg_polarity': avg_polarity,
        'avg_subjectivity': avg_subjectivity
    }

# 2. RESPONSE TIME ANALYSIS
def response_time_analysis(selected_user, df):
    """Analyze response times between messages"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = df[df['user'] != 'group_notification'].copy()
    df = df.sort_values('date')
    
    response_times = []
    responding_users = []
    initiating_users = []
    
    for i in range(1, len(df)):
        current_user = df.iloc[i]['user']
        prev_user = df.iloc[i-1]['user']
        
        if current_user != prev_user:  # Different user responding
            time_diff = (df.iloc[i]['date'] - df.iloc[i-1]['date']).total_seconds() / 60  # in minutes
            if time_diff < 1440:  # Less than 24 hours
                response_times.append(time_diff)
                responding_users.append(current_user)
                initiating_users.append(prev_user)
    
    response_df = pd.DataFrame({
        'response_time_minutes': response_times,
        'responding_user': responding_users,
        'initiating_user': initiating_users
    })
    
    avg_response_by_user = response_df.groupby('responding_user')['response_time_minutes'].agg(['mean', 'median', 'count'])
    
    return response_df, avg_response_by_user

def conversation_initiator_analysis(selected_user, df):
    """Find who initiates conversations most often"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = df[df['user'] != 'group_notification'].copy()
    df = df.sort_values('date')
    
    conversation_starters = []
    conversation_gap_threshold = 60  # 1 hour in minutes
    
    for i in range(len(df)):
        if i == 0:
            conversation_starters.append(df.iloc[i]['user'])
        else:
            time_gap = (df.iloc[i]['date'] - df.iloc[i-1]['date']).total_seconds() / 60
            if time_gap > conversation_gap_threshold:
                conversation_starters.append(df.iloc[i]['user'])
    
    initiator_counts = pd.Series(conversation_starters).value_counts()
    return initiator_counts

# 3. MESSAGE LENGTH & COMMUNICATION STYLE ANALYSIS
def message_length_analysis(selected_user, df):
    """Analyze message length patterns"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    temp['message_length'] = temp['message'].str.len()
    temp['word_count'] = temp['message'].str.split().str.len()
    
    length_stats = temp.groupby('user').agg({
        'message_length': ['mean', 'median', 'std'],
        'word_count': ['mean', 'median', 'std']
    }).round(2)
    
    return temp, length_stats

def communication_style_analysis(selected_user, df):
    """Analyze communication styles"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    styles = []
    for _, row in temp.iterrows():
        message = row['message']
        user = row['user']
        
        # Count various style indicators
        exclamation_count = message.count('!')
        question_count = message.count('?')
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message) if len(message) > 0 else 0
        avg_word_length = np.mean([len(word) for word in message.split()]) if message.split() else 0
        
        styles.append({
            'user': user,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': caps_ratio,
            'avg_word_length': avg_word_length,
            'message_length': len(message)
        })
    
    style_df = pd.DataFrame(styles)
    style_summary = style_df.groupby('user').agg({
        'exclamation_count': 'mean',
        'question_count': 'mean',
        'caps_ratio': 'mean',
        'avg_word_length': 'mean',
        'message_length': 'mean'
    }).round(3)
    
    return style_summary

# 4. TOPIC MODELING AND CONTENT ANALYSIS
def topic_modeling(selected_user, df, n_topics=5):
    """Perform topic modeling on conversations"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    messages = temp['message'].tolist()
    
    if len(messages) < 10:  # Not enough data for topic modeling
        return None, None
    
    try:
        # Clean and preprocess text
        try:
            f = open('stop_hinglish.txt', 'r')
            stop_words = f.read().split()
            f.close()
        except FileNotFoundError:
            stop_words = []
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words, ngram_range=(1, 2))
        doc_term_matrix = vectorizer.fit_transform(messages)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': topic.max()
            })
        
        return topics, lda
    except Exception:
        return None, None

def detect_important_moments(selected_user, df):
    """Detect important moments in conversations"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # Keywords that might indicate important moments
    celebration_keywords = ['birthday', 'anniversary', 'congratulations', 'congrats', 'celebration', 'party', 'wedding']
    decision_keywords = ['decide', 'decision', 'choose', 'final', 'confirmed', 'agreed', 'settled']
    emotional_keywords = ['love', 'miss', 'sorry', 'forgive', 'angry', 'upset', 'happy', 'excited']
    
    important_moments = []
    
    for _, row in temp.iterrows():
        message = row['message'].lower()
        score = 0
        moment_type = []
        
        # Check for celebration keywords
        if any(keyword in message for keyword in celebration_keywords):
            score += 2
            moment_type.append('celebration')
        
        # Check for decision keywords
        if any(keyword in message for keyword in decision_keywords):
            score += 1
            moment_type.append('decision')
        
        # Check for emotional keywords
        if any(keyword in message for keyword in emotional_keywords):
            score += 1
            moment_type.append('emotional')
        
        # Check for high engagement (multiple exclamations, caps)
        if message.count('!') > 2 or sum(1 for c in message if c.isupper()) / len(message) > 0.3:
            score += 1
            moment_type.append('high_engagement')
        
        if score >= 2:
            important_moments.append({
                'date': row['date'],
                'user': row['user'],
                'message': row['message'],
                'score': score,
                'types': moment_type
            })
    
    return pd.DataFrame(important_moments).sort_values('score', ascending=False) if important_moments else pd.DataFrame()