from collections import Counter
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud
import re

# ─────────────────────────────────────────────
# STOP WORDS (simple English + Hindi Romanised)
# ─────────────────────────────────────────────
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'i', 'you', 'he', 'she', 'we', 'they', 'this', 'that', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'with', 'from', 'by', 'as', 'or', 'but', 'not', 'what', 'when',
    'where', 'who', 'how', 'all', 'if', 'so', 'my', 'me', 'your', 'our',
    'ok', 'okay', 'ha', 'haha', 'lol', 'yeah', 'yes', 'no', 'hi', 'hello',
    'hey', 'bhai', 'yaar', 'bro', 'kal', 'aaj', 'ab', 'kya', 'nahi',
    'media', 'omitted', 'message', 'deleted', 'null', 'image', 'video',
    'audio', 'sticker', 'gif', 'missed', 'call', 'voice',
}


# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────
def fetch_stats(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = sum(len(m.split()) for m in df['message'])
    num_media = df[df['message'] == '<Media omitted>'].shape[0]
    links = sum(
        len(re.findall(r'http[s]?://\S+', m)) for m in df['message']
    )
    return num_messages, words, num_media, links


# ─────────────────────────────────────────────
# MOST ACTIVE USERS
# ─────────────────────────────────────────────
def most_busy_users(df: pd.DataFrame):
    counts = df['user'].value_counts().head(10)
    percent = round(df['user'].value_counts(normalize=True) * 100, 2).reset_index()
    percent.columns = ['name', 'percent']
    return counts, percent


# ─────────────────────────────────────────────
# WORD CLOUD
# ─────────────────────────────────────────────
def create_wordcloud(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    text = ' '.join(
        word for msg in df['message']
        for word in msg.lower().split()
        if word not in STOP_WORDS and word.isalpha()
    )

    if not text.strip():
        text = "no words available"

    wc = WordCloud(
        width=800, height=400,
        background_color='#0d0d1a',
        colormap='plasma',
        max_words=150,
        prefer_horizontal=0.9,
    ).generate(text)
    return wc


# ─────────────────────────────────────────────
# MOST COMMON WORDS
# ─────────────────────────────────────────────
def most_common_words(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    words = [
        word.lower()
        for msg in df['message']
        for word in msg.split()
        if word.lower() not in STOP_WORDS
        and word.isalpha()
        and len(word) > 2
    ]
    counter = Counter(words)
    return pd.DataFrame(counter.most_common(20), columns=['Word', 'Count'])


# ─────────────────────────────────────────────
# EMOJI ANALYSIS
# ─────────────────────────────────────────────
def emoji_helper(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    all_emojis = [
        ch for msg in df['message']
        for ch in msg
        if emoji.is_emoji(ch)
    ]
    counter = Counter(all_emojis)
    return pd.DataFrame(counter.most_common(15), columns=['Emoji', 'Count'])


# ─────────────────────────────────────────────
# MONTHLY TIMELINE
# ─────────────────────────────────────────────
def monthly_timeline(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = (
        df.groupby(['year', 'month_num', 'month'])
          .count()['message']
          .reset_index()
    )
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline


# ─────────────────────────────────────────────
# DAILY TIMELINE
# ─────────────────────────────────────────────
def daily_timeline(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()


# ─────────────────────────────────────────────
# WEEK ACTIVITY MAP
# ─────────────────────────────────────────────
def week_activity_map(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


# ─────────────────────────────────────────────
# MONTH ACTIVITY MAP
# ─────────────────────────────────────────────
def month_activity_map(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


# ─────────────────────────────────────────────
# ACTIVITY HEATMAP (hour x day)
# ─────────────────────────────────────────────
def activity_heatmap(selected_user: str, df: pd.DataFrame):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    pivot = df.pivot_table(
        index='day_name', columns='period', values='message',
        aggfunc='count'
    ).fillna(0)
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    return pivot
