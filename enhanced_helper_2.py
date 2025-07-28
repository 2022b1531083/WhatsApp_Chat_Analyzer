import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from textblob import TextBlob
from urlextract import URLExtract
import warnings
warnings.filterwarnings('ignore')

extract = URLExtract()

# ==================== GROUP DYNAMICS & SOCIAL NETWORK ANALYSIS ====================

def group_interaction_matrix(df):
    """Create interaction matrix for group chats"""
    # Filter out group notifications and media messages
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')].copy()
    temp = temp.sort_values('date')
    
    users = temp['user'].unique()
    interaction_matrix = pd.DataFrame(0, index=users, columns=users)
    
    # Track who responds to whom
    for i in range(1, len(temp)):
        current_user = temp.iloc[i]['user']
        prev_user = temp.iloc[i-1]['user']
        
        if current_user != prev_user:
            # Check if response is within reasonable time (30 minutes)
            time_diff = (temp.iloc[i]['date'] - temp.iloc[i-1]['date']).total_seconds() / 60
            if time_diff < 30:
                interaction_matrix.loc[current_user, prev_user] += 1
    
    return interaction_matrix

def identify_group_roles(df):
    """Identify different roles users play in group chats"""
    try:
        temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')].copy()
        
        if len(temp) == 0:
            return {}
        
        user_stats = {}
        users = temp['user'].unique()
        
        if len(users) < 2:
            return {}
        
        for user in users:
            user_messages = temp[temp['user'] == user]
            
            # Calculate various metrics
            total_messages = len(user_messages)
            avg_message_length = user_messages['message'].str.len().mean() if total_messages > 0 else 0
            questions_asked = user_messages['message'].str.count('\?').sum()
            exclamations = user_messages['message'].str.count('!').sum()
            
            # Response patterns
            user_indices = temp[temp['user'] == user].index
            responses_to_others = 0
            
            for idx in user_indices:
                if idx > 0 and idx < len(temp):
                    try:
                        prev_user = temp.loc[idx-1, 'user']
                        if prev_user != user:
                            time_diff = (temp.loc[idx, 'date'] - temp.loc[idx-1, 'date']).total_seconds() / 60
                            if time_diff < 30:  # Responded within 30 minutes
                                responses_to_others += 1
                    except (KeyError, IndexError):
                        continue
            
            user_stats[user] = {
                'total_messages': total_messages,
                'avg_message_length': avg_message_length,
                'questions_asked': questions_asked,
                'exclamations': exclamations,
                'responses_to_others': responses_to_others,
                'response_rate': responses_to_others / total_messages if total_messages > 0 else 0
            }
        
        # Classify roles
        roles = {}
        for user, stats in user_stats.items():
            role_scores = {
                'leader': 0,
                'supporter': 0,
                'questioner': 0,
                'lurker': 0,
                'entertainer': 0
            }
            
            # Calculate thresholds safely
            message_counts = temp.groupby('user').size()
            median_messages = message_counts.median() if len(message_counts) > 0 else 0
            q30_messages = message_counts.quantile(0.3) if len(message_counts) > 0 else 0
            q25_messages = message_counts.quantile(0.25) if len(message_counts) > 0 else 0
            avg_message_len = temp['message'].str.len().mean() if len(temp) > 0 else 0
            
            # Leader: High message count, long messages, low response rate (initiates more)
            if stats['total_messages'] > median_messages:
                role_scores['leader'] += 2
            if stats['avg_message_length'] > avg_message_len:
                role_scores['leader'] += 1
            if stats['response_rate'] < 0.5:
                role_scores['leader'] += 1
                
            # Supporter: High response rate, moderate message count
            if stats['response_rate'] > 0.6:
                role_scores['supporter'] += 2
            if stats['total_messages'] > q30_messages:
                role_scores['supporter'] += 1
                
            # Questioner: High question rate
            if stats['questions_asked'] > 0 and stats['total_messages'] > 0:
                role_scores['questioner'] += min(stats['questions_asked'] / stats['total_messages'] * 10, 5)
                
            # Lurker: Low message count
            if stats['total_messages'] < q25_messages:
                role_scores['lurker'] += 3
                
            # Entertainer: High exclamation usage
            if stats['exclamations'] > 0 and stats['total_messages'] > 0:
                role_scores['entertainer'] += min(stats['exclamations'] / stats['total_messages'] * 10, 5)
            
            # Assign primary role
            primary_role = max(role_scores, key=role_scores.get)
            roles[user] = {
                'primary_role': primary_role,
                'role_scores': role_scores,
                'stats': stats
            }
        
        return roles
    except Exception:
        return {}

def communication_bridges_analysis(df):
    """Identify users who bridge conversations between different group members"""
    interaction_matrix = group_interaction_matrix(df)
    
    # Create network graph
    G = nx.from_pandas_adjacency(interaction_matrix, create_using=nx.DiGraph())
    
    # Calculate centrality measures
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        
        bridge_scores = pd.DataFrame({
            'user': list(betweenness.keys()),
            'betweenness': list(betweenness.values()),
            'closeness': list(closeness.values()),
            'eigenvector': list(eigenvector.values())
        })
        
        # Calculate overall bridge score
        bridge_scores['bridge_score'] = (
            bridge_scores['betweenness'] * 0.5 + 
            bridge_scores['closeness'] * 0.3 + 
            bridge_scores['eigenvector'] * 0.2
        )
        
        return bridge_scores.sort_values('bridge_score', ascending=False)
    except Exception:
        return pd.DataFrame()

def group_activity_correlation(df):
    """Analyze how group members' activity affects others"""
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')].copy()
    
    # Group by hour and user to get activity patterns
    hourly_activity = temp.groupby(['date', 'hour', 'user']).size().unstack(fill_value=0)
    
    # Calculate correlation between users' activity
    correlation_matrix = hourly_activity.corr()
    
    # Find strongest correlations (excluding self-correlation)
    strong_correlations = []
    for user1 in correlation_matrix.columns:
        for user2 in correlation_matrix.columns:
            if user1 != user2:
                corr_value = correlation_matrix.loc[user1, user2]
                if abs(corr_value) > 0.3:  # Significant correlation
                    strong_correlations.append({
                        'user1': user1,
                        'user2': user2,
                        'correlation': corr_value,
                        'relationship': 'positive' if corr_value > 0 else 'negative'
                    })
    
    return pd.DataFrame(strong_correlations).sort_values('correlation', key=abs, ascending=False)

# ==================== PREDICTIVE ANALYTICS ====================

def predict_activity_patterns(df):
    """Predict busy/quiet periods based on historical data"""
    temp = df[df['user'] != 'group_notification'].copy()
    
    # Activity by hour and day
    activity_patterns = temp.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
    
    # Calculate average activity for each hour across all days
    avg_hourly_activity = activity_patterns.mean(axis=0)
    
    # Identify peak and low activity hours
    high_activity_threshold = avg_hourly_activity.quantile(0.75)
    low_activity_threshold = avg_hourly_activity.quantile(0.25)
    
    predictions = {
        'peak_hours': avg_hourly_activity[avg_hourly_activity >= high_activity_threshold].index.tolist(),
        'quiet_hours': avg_hourly_activity[avg_hourly_activity <= low_activity_threshold].index.tolist(),
        'hourly_averages': avg_hourly_activity.to_dict()
    }
    
    return predictions

def relationship_evolution_analysis(selected_user, df):
    """Track how communication patterns change over time"""
    if selected_user == 'Overall':
        temp = df[df['user'] != 'group_notification'].copy()
    else:
        temp = df[df['user'] == selected_user].copy()
    
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # Group by month to see evolution
    temp['year_month'] = temp['date'].dt.to_period('M')
    
    monthly_stats = []
    for period in temp['year_month'].unique():
        period_data = temp[temp['year_month'] == period]
        
        # Calculate various metrics for this period
        stats = {
            'period': str(period),
            'message_count': len(period_data),
            'avg_message_length': period_data['message'].str.len().mean(),
            'unique_users': period_data['user'].nunique(),
            'avg_daily_messages': len(period_data) / period_data['only_date'].nunique()
        }
        
        # Sentiment analysis for this period
        try:
            sentiments = []
            for message in period_data['message']:
                blob = TextBlob(message)
                sentiments.append(blob.sentiment.polarity)
            stats['avg_sentiment'] = np.mean(sentiments)
        except Exception:
            stats['avg_sentiment'] = 0
        
        monthly_stats.append(stats)
    
    evolution_df = pd.DataFrame(monthly_stats)
    return evolution_df.sort_values('period')

# ==================== ADVANCED INSIGHTS & REPORTS ====================

def generate_chat_insights(selected_user, df):
    """Generate AI-like insights about the chat"""
    insights = []
    
    # Basic stats
    if selected_user != 'Overall':
        user_df = df[df['user'] == selected_user]
    else:
        user_df = df[df['user'] != 'group_notification']
    
    total_messages = len(user_df)
    unique_users = user_df['user'].nunique()
    
    # Date range
    date_range = (user_df['date'].max() - user_df['date'].min()).days
    
    insights.append(f"📊 This chat contains {total_messages:,} messages from {unique_users} participants over {date_range} days.")
    
    # Activity patterns
    busiest_hour = user_df.groupby('hour').size().idxmax()
    busiest_day = user_df.groupby('day_name').size().idxmax()
    
    insights.append(f"⏰ Most active at {busiest_hour}:00 on {busiest_day}s.")
    
    # Message patterns
    avg_length = user_df['message'].str.len().mean()
    if avg_length > 50:
        insights.append("📝 Participants tend to send longer, detailed messages.")
    else:
        insights.append("💬 Participants prefer short, quick messages.")
    
    # Media usage
    media_count = len(user_df[user_df['message'] == '<Media omitted>\n'])
    media_ratio = media_count / total_messages
    
    if media_ratio > 0.1:
        insights.append(f"📸 High media sharing: {media_ratio:.1%} of messages are media.")
    
    # Sentiment insight
    try:
        from enhanced_helper_1 import sentiment_analysis
        sentiment_data = sentiment_analysis(selected_user, df)
        positive_ratio = len(sentiment_data[sentiment_data['sentiment'] == 'Positive']) / len(sentiment_data)
        if positive_ratio > 0.6:
            insights.append("😊 Overall positive communication tone.")
        elif positive_ratio < 0.3:
            insights.append("😐 Communication tends to be neutral or serious.")
    except Exception:
        pass
    
    # Response time insight
    try:
        from enhanced_helper_1 import response_time_analysis
        response_df, avg_response = response_time_analysis(selected_user, df)
        if not response_df.empty:
            avg_resp_time = response_df['response_time_minutes'].median()
            if avg_resp_time < 10:
                insights.append("⚡ Very responsive conversation - typical response under 10 minutes.")
            elif avg_resp_time < 60:
                insights.append("🕐 Good response time - typically reply within an hour.")
    except Exception:
        pass
    
    return insights

def conversation_highlights(selected_user, df, limit=10):
    """Find the most interesting/important messages"""
    if selected_user != 'Overall':
        temp = df[df['user'] == selected_user]
    else:
        temp = df[df['user'] != 'group_notification']
    
    temp = temp[temp['message'] != '<Media omitted>\n'].copy()
    
    highlights = []
    
    for _, row in temp.iterrows():
        message = row['message']
        score = 0
        
        # Length score (longer messages might be more important)
        if len(message) > 100:
            score += 2
        elif len(message) > 50:
            score += 1
        
        # Emotion score
        exclamations = message.count('!')
        questions = message.count('?')
        score += min(exclamations * 0.5, 2)
        score += min(questions * 0.5, 2)
        
        # Caps usage (excitement/importance)
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message) if len(message) > 0 else 0
        if caps_ratio > 0.2:
            score += 1
        
        # Keywords that might indicate importance
        important_keywords = ['important', 'urgent', 'news', 'announcement', 'update', 'decision', 'final', 'confirmed']
        for keyword in important_keywords:
            if keyword.lower() in message.lower():
                score += 1
        
        highlights.append({
            'date': row['date'],
            'user': row['user'],
            'message': message,
            'score': score
        })
    
    highlights_df = pd.DataFrame(highlights)
    return highlights_df.nlargest(limit, 'score') if not highlights_df.empty else pd.DataFrame()

# ==================== GAMIFICATION & ACHIEVEMENT SYSTEM ====================

def calculate_communication_badges(selected_user, df):
    """Award badges based on communication patterns"""
    if selected_user != 'Overall':
        user_df = df[df['user'] == selected_user]
    else:
        user_df = df[df['user'] != 'group_notification']
    
    badges = {}
    
    for user in user_df['user'].unique():
        user_messages = user_df[user_df['user'] == user]
        user_badges = []
        
        # Message count badges
        msg_count = len(user_messages)
        if msg_count >= 10000:
            user_badges.append("🏆 Chat Legend (10K+ messages)")
        elif msg_count >= 5000:
            user_badges.append("💎 Super Communicator (5K+ messages)")
        elif msg_count >= 1000:
            user_badges.append("🌟 Active Chatter (1K+ messages)")
        elif msg_count >= 100:
            user_badges.append("💬 Regular User (100+ messages)")
        
        # Time-based badges
        date_range = (user_messages['date'].max() - user_messages['date'].min()).days
        if date_range >= 365:
            user_badges.append("📅 Long-term Friend (1+ year)")
        elif date_range >= 180:
            user_badges.append("🗓️ Consistent Chatter (6+ months)")
        
        # Activity badges
        unique_days = user_messages['only_date'].nunique()
        if unique_days >= 100:
            user_badges.append("⏰ Daily Communicator (100+ active days)")
        
        # Media badges
        media_count = len(user_messages[user_messages['message'] == '<Media omitted>\n'])
        if media_count >= 500:
            user_badges.append("📸 Media Master (500+ media)")
        elif media_count >= 100:
            user_badges.append("🖼️ Photo Sharer (100+ media)")
        
        # Time-specific badges
        night_messages = len(user_messages[user_messages['hour'].between(22, 23) | user_messages['hour'].between(0, 5)])
        if night_messages >= 100:
            user_badges.append("🌙 Night Owl (100+ late night messages)")
        
        early_messages = len(user_messages[user_messages['hour'].between(5, 8)])
        if early_messages >= 100:
            user_badges.append("🌅 Early Bird (100+ early morning messages)")
        
        # Response badges
        try:
            from enhanced_helper_1 import response_time_analysis
            response_df, avg_response = response_time_analysis(user, df)
            if not response_df.empty and user in avg_response.index:
                avg_resp_time = avg_response.loc[user, 'mean']
                if avg_resp_time < 5:
                    user_badges.append("⚡ Lightning Fast (Avg <5min response)")
                elif avg_resp_time < 30:
                    user_badges.append("🏃 Quick Responder (Avg <30min response)")
        except Exception:
            pass
        
        badges[user] = user_badges
    
    return badges

def personality_matching_analysis(df):
    """Compare communication styles between users"""
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')].copy()
    
    user_personalities = {}
    
    for user in temp['user'].unique():
        user_messages = temp[temp['user'] == user]
        
        # Calculate personality metrics
        total_messages = len(user_messages)
        avg_length = user_messages['message'].str.len().mean()
        question_ratio = user_messages['message'].str.count('\?').sum() / total_messages
        exclamation_ratio = user_messages['message'].str.count('!').sum() / total_messages
        caps_usage = user_messages['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0).mean()
        
        # Sentiment analysis
        try:
            sentiments = []
            for message in user_messages['message']:
                blob = TextBlob(message)
                sentiments.append(blob.sentiment.polarity)
            avg_sentiment = np.mean(sentiments)
        except Exception:
            avg_sentiment = 0
        
        # Time patterns
        night_ratio = len(user_messages[user_messages['hour'].between(22, 23) | user_messages['hour'].between(0, 5)]) / total_messages
        weekend_ratio = len(user_messages[user_messages['day_name'].isin(['Saturday', 'Sunday'])]) / total_messages
        
        user_personalities[user] = {
            'avg_message_length': avg_length,
            'question_ratio': question_ratio,
            'exclamation_ratio': exclamation_ratio,
            'caps_usage': caps_usage,
            'avg_sentiment': avg_sentiment,
            'night_ratio': night_ratio,
            'weekend_ratio': weekend_ratio
        }
    
    # Calculate similarity between users
    similarities = []
    users = list(user_personalities.keys())
    
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            user1, user2 = users[i], users[j]
            
            # Calculate Euclidean distance between personality vectors
            p1 = np.array(list(user_personalities[user1].values()))
            p2 = np.array(list(user_personalities[user2].values()))
            
            # Normalize to prevent length bias
            p1_norm = (p1 - np.mean(p1)) / (np.std(p1) + 1e-8)
            p2_norm = (p2 - np.mean(p2)) / (np.std(p2) + 1e-8)
            
            similarity = 1 / (1 + np.linalg.norm(p1_norm - p2_norm))
            
            similarities.append({
                'user1': user1,
                'user2': user2,
                'similarity_score': similarity,
                'compatibility': 'High' if similarity > 0.7 else 'Medium' if similarity > 0.5 else 'Low'
            })
    
    return pd.DataFrame(similarities).sort_values('similarity_score', ascending=False), user_personalities

# ==================== PRIVACY & SECURITY FEATURES ====================

def create_anonymous_analysis(df):
    """Create anonymized version of data for privacy-conscious analysis"""
    anonymous_df = df.copy()
    
    # Replace usernames with generic identifiers
    unique_users = [user for user in df['user'].unique() if user != 'group_notification']
    user_mapping = {user: f"User_{i+1}" for i, user in enumerate(unique_users)}
    user_mapping['group_notification'] = 'group_notification'
    
    anonymous_df['user'] = anonymous_df['user'].map(user_mapping)
    
    # Remove actual message content, keep only metadata
    anonymous_df['message_length'] = anonymous_df['message'].str.len()
    anonymous_df['word_count'] = anonymous_df['message'].str.split().str.len()
    anonymous_df['has_media'] = (anonymous_df['message'] == '<Media omitted>\n').astype(int)
    anonymous_df['has_link'] = anonymous_df['message'].apply(lambda x: 1 if extract.find_urls(x) else 0)
    anonymous_df['question_marks'] = anonymous_df['message'].str.count('\?')
    anonymous_df['exclamations'] = anonymous_df['message'].str.count('!')
    
    # Remove actual message content
    anonymous_df = anonymous_df.drop('message', axis=1)
    
    return anonymous_df, user_mapping

# ==================== ADVANCED VISUALIZATIONS DATA PREP ====================

def prepare_network_graph_data(df):
    """Prepare data for network visualization"""
    interaction_matrix = group_interaction_matrix(df)
    
    # Create nodes and edges for visualization
    nodes = []
    for user in interaction_matrix.index:
        total_interactions = interaction_matrix.loc[user].sum() + interaction_matrix[user].sum()
        nodes.append({
            'id': user,
            'label': user,
            'size': min(total_interactions * 2, 50),  # Scale node size
            'interactions': total_interactions
        })
    
    edges = []
    for user1 in interaction_matrix.index:
        for user2 in interaction_matrix.columns:
            weight = interaction_matrix.loc[user1, user2]
            if weight > 0:
                edges.append({
                    'source': user1,
                    'target': user2,
                    'weight': weight,
                    'width': min(weight * 0.5, 10)  # Scale edge width
                })
    
    return {'nodes': nodes, 'edges': edges}

def prepare_sentiment_heatmap_data(selected_user, df):
    """Prepare sentiment data for heatmap visualization"""
    from enhanced_helper_1 import sentiment_analysis
    sentiment_df = sentiment_analysis(selected_user, df)
    
    # Create sentiment score by day and hour
    sentiment_df['sentiment_score'] = sentiment_df['polarity']
    heatmap_data = sentiment_df.groupby(['day_name', 'hour'])['sentiment_score'].mean().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order, fill_value=0)
    
    return heatmap_data

def conversation_flow_analysis(selected_user, df):
    """Analyze conversation flow patterns"""
    if selected_user != 'Overall':
        temp = df[df['user'] == selected_user]
    else:
        temp = df[df['user'] != 'group_notification']
    
    temp = temp[temp['message'] != '<Media omitted>\n'].copy()
    temp = temp.sort_values('date')
    
    flow_patterns = []
    
    for i in range(1, len(temp)):
        prev_user = temp.iloc[i-1]['user']
        curr_user = temp.iloc[i]['user']
        time_gap = (temp.iloc[i]['date'] - temp.iloc[i-1]['date']).total_seconds() / 60
        
        flow_patterns.append({
            'from_user': prev_user,
            'to_user': curr_user,
            'time_gap_minutes': time_gap,
            'is_continuation': prev_user == curr_user,
            'hour': temp.iloc[i]['hour'],
            'day_name': temp.iloc[i]['day_name']
        })
    
    flow_df = pd.DataFrame(flow_patterns)
    
    # Analyze patterns
    avg_response_time = flow_df[flow_df['is_continuation'] == False]['time_gap_minutes'].median()
    continuation_ratio = flow_df['is_continuation'].mean()
    
    return {
        'flow_data': flow_df,
        'avg_response_time': avg_response_time,
        'continuation_ratio': continuation_ratio
    }

# ==================== EXPORT AND REPORTING FUNCTIONS ====================

def generate_comprehensive_report(selected_user, df):
    """Generate a comprehensive analysis report"""
    report = {
        'basic_stats': {},
        'sentiment_analysis': {},
        'response_patterns': {},
        'communication_style': {},
        'important_moments': {},
        'insights': [],
        'badges': {},
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Basic statistics
        from enhanced_helper_1 import fetch_stats
        num_messages, words, num_media_messages, num_links = fetch_stats(selected_user, df)
        report['basic_stats'] = {
            'total_messages': num_messages,
            'total_words': words,
            'media_messages': num_media_messages,
            'links_shared': num_links
        }
        
        # Sentiment analysis
        try:
            from enhanced_helper_1 import sentiment_analysis
            sentiment_data = sentiment_analysis(selected_user, df)
            if not sentiment_data.empty:
                sentiment_counts = sentiment_data['sentiment'].value_counts()
                report['sentiment_analysis'] = {
                    'positive_ratio': sentiment_counts.get('Positive', 0) / len(sentiment_data),
                    'negative_ratio': sentiment_counts.get('Negative', 0) / len(sentiment_data),
                    'neutral_ratio': sentiment_counts.get('Neutral', 0) / len(sentiment_data),
                    'avg_polarity': sentiment_data['polarity'].mean()
                }
        except Exception:
            pass
        
        # Response patterns
        try:
            from enhanced_helper_1 import response_time_analysis
            response_df, avg_response = response_time_analysis(selected_user, df)
            if not response_df.empty:
                report['response_patterns'] = {
                    'avg_response_time_minutes': response_df['response_time_minutes'].mean(),
                    'median_response_time_minutes': response_df['response_time_minutes'].median(),
                    'fastest_responders': avg_response.sort_values('mean').head(3).to_dict()
                }
        except Exception:
            pass
        
        # Communication style
        try:
            from enhanced_helper_1 import communication_style_analysis
            style_analysis = communication_style_analysis(selected_user, df)
            report['communication_style'] = style_analysis.to_dict()
        except Exception:
            pass
        
        # Important moments
        try:
            from enhanced_helper_1 import detect_important_moments
            moments = detect_important_moments(selected_user, df)
            report['important_moments'] = {
                'count': len(moments),
                'top_moments': moments.head(5).to_dict('records') if not moments.empty else []
            }
        except Exception:
            pass
        
        # Generate insights
        report['insights'] = generate_chat_insights(selected_user, df)
        
        # Badges
        try:
            badges = calculate_communication_badges(selected_user, df)
            report['badges'] = badges
        except Exception:
            pass
        
    except Exception as e:
        report['error'] = str(e)
    
    return report