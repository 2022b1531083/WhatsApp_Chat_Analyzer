import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import re
from collections import Counter
import emoji
from textblob import TextBlob
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize URL extractor
extract = URLExtract()

# Download necessary NLTK data (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Helper functions
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    
    num_media_messages = df[df['is_media'] == True].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    num_links = len(links)
    total_words = len(words)
    
    return num_messages, total_words, num_media_messages, num_links

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_user = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x, df_user

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
        
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
        
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    
    return user_heatmap

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    
    return df_wc

def remove_stop_words(message):
    y = []
    stopwords = nltk.corpus.stopwords.words('english')
    for word in message.lower().split():
        if word not in stopwords and word not in string.punctuation:
            y.append(word)
    return " ".join(y)

def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['is_media'] == False]
    
    words = []
    for message in temp['message']:
        words.extend(message.split())

    stop_words = nltk.corpus.stopwords.words('english')
    
    words_filtered = [word.lower() for word in words if word.lower() not in stop_words and word.lower() != '<media' and word.lower() != 'omitted>']
    
    most_common_df = pd.DataFrame(Counter(words_filtered).most_common(20))
    
    most_common_df.columns = ['Word', 'Count']
    
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(emojis)))
    emoji_df.columns = ['emoji', 'count']
    
    return emoji_df

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiments = []
    for message in df['message']:
        blob = TextBlob(message)
        if blob.sentiment.polarity > 0.1:
            sentiments.append('Positive')
        elif blob.sentiment.polarity < -0.1:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
            
    sentiment_df = pd.DataFrame({'message': df['message'], 'sentiment': sentiments})
    
    return sentiment_df

# === START OF ADDED FUNCTIONS ===

def response_time_analysis(selected_user, df):
    """Calculates average response time between users."""
    if selected_user == 'Overall':
        temp_df = df[df['user'] != 'group_notification'].copy()
        temp_df['prev_user'] = temp_df['user'].shift(1)
        temp_df['prev_date'] = temp_df['date'].shift(1)
        
        response_times = []
        for i, row in temp_df.iterrows():
            if i > 0 and row['user'] != row['prev_user']:
                time_diff = (row['date'] - row['prev_date']).total_seconds() / 60
                if time_diff > 0:
                    response_times.append({
                        'user': row['user'],
                        'response_time_minutes': time_diff
                    })
        
        if not response_times:
            return pd.DataFrame(), pd.DataFrame()
            
        response_df = pd.DataFrame(response_times)
        avg_response_by_user = response_df.groupby('user')['response_time_minutes'].mean().reset_index()
        avg_response_by_user.columns = ['User', 'Avg Response Time (min)']
        return response_df, avg_response_by_user
    else:
        return pd.DataFrame(), pd.DataFrame()

def message_length_analysis(selected_user, df):
    """Analyzes message length and word count per user."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp_df = df[df['user'] != 'group_notification'].copy()
    temp_df['message_length'] = temp_df['message'].apply(lambda x: len(x))
    temp_df['word_count'] = temp_df['message'].apply(lambda x: len(x.split()))
    
    avg_length_stats = temp_df.groupby('user')[['message_length', 'word_count']].mean().reset_index()
    avg_length_stats.columns = ['User', 'Avg Message Length', 'Avg Word Count']
    
    return temp_df, avg_length_stats

def topic_modeling(selected_user, df):
    """Performs Topic Modeling using LDA on chat messages."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp_df = df[(df['user'] != 'group_notification') & (df['is_media'] == False)].copy()
    
    if len(temp_df) < 10:
        return None, None
    
    # Preprocess text
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['media', 'omitted', 'deleted', 'this', 'that', 'with', 'from', 'for'])
    
    temp_df['cleaned_message'] = temp_df['message'].apply(lambda x: ' '.join([word.lower() for word in str(x).split() if word.lower() not in stop_words and len(word)>2]))
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
    try:
        tfidf_matrix = vectorizer.fit_transform(temp_df['cleaned_message'])
    except ValueError:
        return [], None
        
    # LDA model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics_list = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics_list.append({
            'topic_id': topic_idx,
            'words': top_words
        })
        
    return topics_list, lda