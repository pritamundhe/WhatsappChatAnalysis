# 💬 WhatsApp Chat Analyzer — ML-Powered Dashboard

> A production-quality chat analytics application with 5 integrated ML models.
> Built with Python · Streamlit · Pandas · scikit-learn · VADER · TextBlob · Gensim

---

## ✨ Features

### 📊 Core Analytics
- **Message statistics** — total messages, words, media, links
- **Monthly & daily timelines** — interactive Plotly area/line charts
- **Activity heatmap** — hour × day message frequency
- **Busiest days & months** — bar charts with vibrant gradients
- **Most active users** — horizontal bar + donut pie chart

### 🔤 Text Analysis
- **WordCloud** — styled with plasma colormap on dark background
- **Top 20 Words** — frequency bar chart (stop-words filtered)
- **Emoji Analysis** — pie chart + ranked table

### 🤖 ML Models (Resume-Worthy!)

| Model | Algorithm | Resume Point |
|-------|-----------|--------------|
| **Sentiment Analysis** | VADER + TextBlob ensemble | NLP, ensemble methods |
| **Topic Modeling** | LDA (Latent Dirichlet Allocation) | Unsupervised ML, NLP |
| **Behavior Prediction** | Random Forest Classifier | Supervised ML, classification |
| **Spam Detection** | TF-IDF + Logistic Regression | Feature engineering, classification |
| **Chat Summarization** | Extractive NLP (frequency-based) | NLP, text summarization |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Upload your chat
- Open WhatsApp → Any Chat → **⋮ More → Export Chat** (without media)
- Upload the `.txt` file in the sidebar
- Select a user and click **Analyse Now**

---

## 📁 Project Structure

```
WhatsappChatAnalysis/
├── app.py              # Main Streamlit dashboard (vibrant colorful UI)
├── preprocessor.py     # WhatsApp .txt parser (12h & 24h format support)
├── helper.py           # Analytics & visualization helpers
├── ml_models.py        # 5 ML models (sentiment, topics, behavior, spam, summary)
├── requirements.txt    # Python dependencies
├── sample_chat.txt     # Demo chat for testing
└── README.md           # This file
```

---

## 🎨 Tech Stack

- **Frontend**: Streamlit + Custom CSS (glassmorphism, gradients, animations)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **NLP**: NLTK, VADER, TextBlob, WordCloud
- **ML**: scikit-learn (LDA, Random Forest, TF-IDF, Logistic Regression)

---

## 📝 Resume Description

> **WhatsApp Chat Analysis System** | Python · Streamlit · scikit-learn · NLP  
> Built an end-to-end chat analytics dashboard with 5 integrated ML models:
> - **Sentiment Analysis** using VADER + TextBlob ensemble scoring
> - **Topic Modeling** with LDA to extract conversation themes
> - **User Behavior Prediction** using Random Forest classifier (peak hour detection)
> - **Spam/Abusive Message Detection** using TF-IDF + Logistic Regression pipeline
> - **Extractive Chat Summarization** using NLP frequency-based scoring
> 
> Deployed via Streamlit with custom CSS dark-themed UI. Supports 12h/24h chat formats.

---

## 📜 License

MIT License — Free to use, modify and distribute.
