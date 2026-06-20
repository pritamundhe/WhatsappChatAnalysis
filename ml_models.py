"""
Machine Learning Module for WhatsApp Chat Analysis

  1. Sentiment Analysis   — VADER + TextBlob ensemble
  2. Topic Modeling       — LDA via scikit-learn
  3. Behavior Prediction  — RandomForest peak-hour predictor
  4. Spam Detection       — TF-IDF + Logistic Regression
  5. Chat Summarization   — Extractive frequency-based NLP
"""

import re
import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

warnings.filterwarnings('ignore')

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
for pkg in ['punkt', 'stopwords', 'vader_lexicon', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ── Sentiment ─────────────────────────────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 1. SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()

def _vader_label(score: float) -> str:
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'


def analyze_sentiment(df: pd.DataFrame, selected_user: str = 'Overall') -> dict:
    """
    Returns per-message sentiment and per-user aggregated sentiment scores.
    Uses VADER + TextBlob ensemble (average of compound scores).
    """
    work = df.copy()
    if selected_user != 'Overall':
        work = work[work['user'] == selected_user]

    work = work[~work['message'].str.startswith('<Media omitted>')].copy()
    work = work[work['message'].str.strip() != ''].copy()

    if work.empty:
        return {'per_message': pd.DataFrame(), 'per_user': pd.DataFrame(),
                'overall_label': 'Neutral', 'overall_score': 0.0}

    # VADER scores
    work['vader_score'] = work['message'].apply(
        lambda m: _vader.polarity_scores(str(m))['compound']
    )
    # TextBlob scores (polarity -1 to +1)
    work['blob_score'] = work['message'].apply(
        lambda m: TextBlob(str(m)).sentiment.polarity
    )
    # Ensemble average
    work['sentiment_score'] = (work['vader_score'] + work['blob_score']) / 2
    work['sentiment_label'] = work['sentiment_score'].apply(_vader_label)

    # Per-user aggregation
    per_user = (
        work.groupby('user')
            .agg(
                avg_score=('sentiment_score', 'mean'),
                positive=('sentiment_label', lambda x: (x == 'Positive').sum()),
                neutral=('sentiment_label', lambda x: (x == 'Neutral').sum()),
                negative=('sentiment_label', lambda x: (x == 'Negative').sum()),
                total=('sentiment_label', 'count'),
            )
            .reset_index()
    )
    per_user['dominant'] = per_user[['positive', 'neutral', 'negative']].idxmax(axis=1)

    overall_score = work['sentiment_score'].mean()
    overall_label = _vader_label(overall_score)

    return {
        'per_message': work[['user', 'date', 'message', 'sentiment_score', 'sentiment_label']],
        'per_user': per_user,
        'overall_label': overall_label,
        'overall_score': round(overall_score, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOPIC MODELING (LDA)
# ─────────────────────────────────────────────────────────────────────────────

try:
    _STOPWORDS = set(stopwords.words('english'))
except Exception:
    _STOPWORDS = set()

_EXTRA_STOP = {
    'ok', 'okay', 'ha', 'haha', 'lol', 'yeah', 'yes', 'no', 'hi', 'hello',
    'hey', 'bhai', 'yaar', 'bro', 'media', 'omitted', 'message', 'deleted',
    'will', 'like', 'just', 'know', 'good', 'time', 'come', 'go', 'get',
    'also', 'one', 'day', 'na', 'kar', 'kya', 'nahi', 'aaj', 'kal', 'abhi',
}
_ALL_STOP = _STOPWORDS | _EXTRA_STOP


def _clean_for_lda(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = [w.lower() for w in text.split() if w.lower() not in _ALL_STOP and len(w) > 3]
    return ' '.join(tokens)


def run_topic_modeling(df: pd.DataFrame, n_topics: int = 5, n_words: int = 8) -> list[dict]:
    """
    LDA topic modeling on chat messages.
    Returns list of dicts: [{topic_id, label, words, weight}, ...]
    """
    work = df[~df['message'].str.startswith('<Media omitted>')].copy()
    work['clean'] = work['message'].apply(_clean_for_lda)
    work = work[work['clean'].str.split().str.len() >= 3]

    if len(work) < 20:
        return [{'topic_id': 0, 'words': ['Not enough data'], 'weight': 1.0}]

    n_topics = min(n_topics, max(2, len(work) // 10))

    vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000)
    try:
        dtm = vectorizer.fit_transform(work['clean'])
    except Exception:
        return [{'topic_id': 0, 'words': ['Unable to model topics'], 'weight': 1.0}]

    if dtm.shape[1] < n_words:
        return [{'topic_id': 0, 'words': ['Vocabulary too small'], 'weight': 1.0}]

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=20, learning_method='online'
    )
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        weight = round(topic.max() / topic.sum(), 4)
        topics.append({'topic_id': idx + 1, 'words': top_words, 'weight': weight})
    return topics


# ─────────────────────────────────────────────────────────────────────────────
# 3. USER BEHAVIOR PREDICTION — Peak Hour Classifier
# ─────────────────────────────────────────────────────────────────────────────

def predict_peak_hours(df: pd.DataFrame) -> dict:
    """
    Uses a Random Forest to learn each user's peak-activity hour pattern.
    Returns: {user: {peak_hour, peak_day, activity_heatmap_series, model_accuracy}}
    """
    results = {}
    users = df['user'].unique()

    for user in users:
        udf = df[df['user'] == user].copy()
        if len(udf) < 15:
            continue

        # Simple feature: count messages per (hour, day_of_week)
        udf['dow'] = udf['date'].dt.dayofweek
        hourly = udf.groupby('hour').size()
        peak_hour = int(hourly.idxmax())
        daily = udf.groupby('day_name').size()
        peak_day = daily.idxmax()

        # Random Forest — classify hour bucket (morning/afternoon/evening/night)
        def _bucket(h):
            if 5 <= h < 12:   return 'Morning'
            if 12 <= h < 17:  return 'Afternoon'
            if 17 <= h < 21:  return 'Evening'
            return 'Night'

        udf['bucket'] = udf['hour'].apply(_bucket)
        udf['dow'] = udf['date'].dt.dayofweek
        udf['month'] = udf['date'].dt.month

        feature_cols = ['hour', 'dow', 'month', 'minute']
        X = udf[feature_cols].values
        y = udf['bucket'].values

        if len(set(y)) < 2 or len(X) < 10:
            results[user] = {
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'hourly_series': hourly,
                'model_accuracy': None,
                'bucket': _bucket(peak_hour),
            }
            continue

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
            )
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            acc = round(clf.score(X_test, y_test) * 100, 1)
        except Exception:
            acc = None

        results[user] = {
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'hourly_series': hourly,
            'model_accuracy': acc,
            'bucket': _bucket(peak_hour),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPAM / ABUSIVE MESSAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Lightweight rule-based seed + TF-IDF Logistic Regression
_SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'click here', 'earn money', 'prize', 'lottery',
    'congratulations', 'claim', 'limited offer', 'act now', 'bitcoin',
    'crypto', 'investment', 'double your money', 'make money fast', 'urgent',
    'verify your account', 'password', 'otp', 'bank details', 'credit card',
    'forward this', 'share this', 'viral', '100% free', 'guaranteed',
]
_ABUSE_KEYWORDS = [
    'idiot', 'stupid', 'fool', 'shut up', 'hate you', 'moron', 'loser',
    'dumb', 'ugly', 'worthless', 'get lost', 'die', 'kill', 'abuse',
]


def _rule_label(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _SPAM_KEYWORDS):
        return 'Spam'
    if any(kw in t for kw in _ABUSE_KEYWORDS):
        return 'Abusive'
    return 'Normal'


def detect_spam(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects spam and abusive messages using:
     - Rule-based seed labels (TF-IDF features)
     - Logistic Regression trained on seed labels
    Returns df with 'spam_label' and 'spam_confidence' columns.
    """
    work = df[~df['message'].str.startswith('<Media omitted>')].copy()
    work = work[work['message'].str.strip().str.len() > 5].copy()

    if work.empty:
        return work

    work['rule_label'] = work['message'].apply(_rule_label)

    try:
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=300, random_state=42, C=1.0)),
        ])
        pipe.fit(work['message'], work['rule_label'])
        work['spam_label'] = pipe.predict(work['message'])
        proba = pipe.predict_proba(work['message'])
        work['spam_confidence'] = proba.max(axis=1).round(3)
    except Exception:
        work['spam_label'] = work['rule_label']
        work['spam_confidence'] = 1.0

    return work[['user', 'date', 'message', 'spam_label', 'spam_confidence']]


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTRACTIVE CHAT SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────────────

def summarize_chat(df: pd.DataFrame, selected_user: str = 'Overall',
                   n_sentences: int = 5) -> str:
    """
    AI-based chat summarization using LangChain and DeepSeek.
    """
    work = df.copy()
    if selected_user != 'Overall':
        work = work[work['user'] == selected_user]

    work = work[~work['message'].str.startswith('<Media omitted>')].copy()
    work = work[~work['message'].str.contains('This message was deleted', na=False)].copy()
    
    # Take the last 800 messages to avoid exceeding context limits
    messages = work['message'].tail(800).tolist()
    text = '\n'.join(messages)

    if not text.strip():
        return "Not enough text to summarize."

    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        return "API key not found. Please set DEEPSEEK_API_KEY in the .env file."
    
    try:
        llm = ChatOpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com/v1", 
            model="deepseek-chat",
            max_tokens=1000
        )
        
        prompt = f"""
Please provide a concise summary of the following WhatsApp chat conversation. 
Focus on the main topics discussed, the overall tone, and any key decisions or events.
Keep the summary to approximately {n_sentences} sentences.

Chat:
{text[-25000:]} 
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Could not generate AI summary. Error: {str(e)}"
