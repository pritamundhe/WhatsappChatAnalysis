"""
app.py — WhatsApp Chat Analysis Dashboard
==========================================
Vibrant, colorful Streamlit UI with ML-powered insights.
Run: streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from io import BytesIO

import preprocessor
import helper
import ml_models

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

.material-symbols-outlined {
  vertical-align: middle;
  margin-right: 8px;
  font-size: inherit;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 18px 22px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem; font-weight: 500; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.8rem; font-weight: 700; }

/* ── Section Headers ── */
.section-header {
    color: #f8fafc;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}
.sub-header {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.2rem 0 0.8rem;
}

/* ── ML Badge ── */
.ml-badge {
    display: inline-block;
    background-color: #312e81;
    color: #818cf8;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Sentiment Card ── */
.sentiment-card {
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    border: 1px solid #334155;
    background-color: #1e293b;
}
.sentiment-positive { border-left: 4px solid #22c55e; }
.sentiment-neutral  { border-left: 4px solid #eab308; }
.sentiment-negative { border-left: 4px solid #ef4444; }

/* ── Topic Card ── */
.topic-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}
.topic-number {
    background-color: #4f46e5;
    color: white;
    border-radius: 4px;
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    margin-right: 10px;
}

/* ── Pill Tags ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px;
}
.pill-purple { background: #4c1d95; color: #d8b4fe; border: 1px solid #5b21b6; }
.pill-blue   { background: #1e3a8a; color: #93c5fd; border: 1px solid #1e40af; }
.pill-green  { background: #14532d; color: #86efac; border: 1px solid #166534; }
.pill-red    { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
.pill-yellow { background: #713f12; color: #fde047; border: 1px solid #854d0e; }

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background-color: #334155;
    margin: 2.5rem 0;
    border: none;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover {
    background-color: #6366f1 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #334155;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid #334155;
    gap: 20px;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500;
    padding-bottom: 10px;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
    background: transparent !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploadDropzone"] {
    background-color: #1e293b !important;
    border: 2px dashed #4f46e5 !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #818cf8 !important;
    background-color: #1e293b !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}
[data-testid="stFileUploadDropzone"] * {
    color: #cbd5e1 !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background-color: #6366f1 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0f172a',
    'axes.facecolor':    '#0f172a',
    'axes.edgecolor':    '#334155',
    'axes.labelcolor':   '#cbd5e1',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'text.color':        '#f1f5f9',
    'grid.color':        '#1e293b',
    'grid.linestyle':    '-',
    'grid.alpha':        1.0,
    'font.family':       'sans-serif',
})
GRAD_COLORS = ['#6366f1', '#8b5cf6', '#0ea5e9', '#06b6d4', '#14b8a6', '#22c55e', '#84cc16', '#eab308', '#f59e0b', '#f97316']

def _fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem; color:#4f46e5;'><span class=material-symbols-outlined style='font-size:3rem;'>chat</span></div>
        <div style='font-size:1.3rem; font-weight:800; background-color:transparent;
             color:#f8fafc;'>
             WhatsApp Analyzer
        </div>
        <div style='font-size:0.75rem; color:#cbd5e1; margin-top:4px;'>ML-Powered Chat Intelligence</div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.3); margin:1rem 0;'>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Chat Export (.txt)",
        type='txt',
        help="Export from WhatsApp → ⋮ → More → Export Chat (without media)"
    )




# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER (no file uploaded)
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    hero_html = """
    <div style='text-align:center; padding: 1rem 2rem 5rem;'>
        <div style='display:inline-flex; align-items:center; justify-content:center; width:80px; height:80px; border-radius:50%; background-color:#1e293b; border:1px solid #334155; margin-bottom:1.5rem;'>
            <span class=material-symbols-outlined style='font-size:2.5rem; color:#818cf8; margin:0;'>chat</span>
        </div>
        <h1 style='font-size:3rem; font-weight:800; color:#f8fafc; margin-bottom:0.5rem; letter-spacing:-0.02em;'>
            WhatsApp Chat Analyzer
        </h1>
        <p style='font-size:1.15rem; color:#94a3b8; max-width:650px; margin:0 auto 2.5rem; line-height:1.6;'>
            Unlock powerful AI & ML insights from your WhatsApp conversations —
            sentiment, topics, behavior patterns, spam detection & more.
        </p>
        <div style='display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-bottom:3.5rem;'>
    """

    for badge, color in [
        ("<span class=material-symbols-outlined>psychology</span> Sentiment AI", "#818cf8"), ("<span class=material-symbols-outlined>library_books</span> Topic Modeling", "#a78bfa"),
        ("<span class=material-symbols-outlined>schedule</span> Behavior Prediction", "#c084fc"), ("<span class=material-symbols-outlined>warning</span> Spam Detection", "#e879f9"),
        ("<span class=material-symbols-outlined>edit_document</span> NLP Summarization", "#f472b6"),
    ]:
        hero_html += f"""
        <span style='background-color:#1e293b; border:1px solid #334155;
            color:{color}; padding:8px 20px; border-radius:6px; font-weight:500; font-size:0.9rem;'>
            {badge}
        </span>"""

    hero_html += """
        </div>
        <div style='background-color:#1e293b;
            border:1px solid #334155; border-radius:12px; padding:2rem; max-width:550px;
            margin:0 auto; text-align:left; box-shadow:0 4px 6px -1px rgba(0,0,0,0.5);'>
            <div style='font-size:1.2rem; font-weight:600; color:#f8fafc; margin-bottom:1rem; display:flex; align-items:center;'>
                <span class=material-symbols-outlined style='color:#818cf8;'>upload</span> 
                <span style='margin-left:8px;'>How to upload your chat</span>
            </div>
            <div style='color:#cbd5e1; font-size:0.95rem; line-height:1.6;'>
                <ol style='margin:0; padding-left:1.5rem;'>
                    <li>Open <strong>WhatsApp</strong> on your phone</li>
                    <li>Go to the chat you want to analyze</li>
                    <li>Tap <strong>⋮ More</strong> > <strong>Export Chat</strong> (Without media)</li>
                    <li>Upload the exported <code>.txt</code> file in the sidebar</li>
                </ol>
            </div>
        </div>
    </div>
    """
    st.html(hero_html)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS FILE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Parsing chat…"):
    raw = uploaded_file.read().decode('utf-8', errors='replace')
    try:
        df = preprocessor.preprocess(raw)
    except Exception as e:
        st.error(f"Could not parse this file. Error: {e}")
        st.stop()

if df.empty:
    st.warning("No messages found. Ensure the file is a valid WhatsApp export.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# USER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

with st.sidebar:
    st.markdown("<hr style='border-color:rgba(99,102,241,0.3);'>", unsafe_allow_html=True)
    user_list = ['Overall'] + sorted(df['user'].unique().tolist())
    
    # If the user changes the selectbox, we reset the analysis state so they have to click analyze again
    # or we can just let it auto-update. Auto-update is usually better, but let's keep the explicit button behavior.
    def on_user_change():
        st.session_state.analyzed = False
        
    selected_user = st.selectbox("Select User", user_list, on_change=on_user_change)
    analyse_btn = st.button("Analyse Now", use_container_width=True)

if analyse_btn:
    st.session_state.analyzed = True

if not st.session_state.analyzed:
    st.markdown("""
    <div style='text-align:center; padding:3rem 3rem 1rem 3rem; color:#94a3b8;'>
        <div style='font-size:3rem;'><span class=material-symbols-outlined>check_circle</span></div>
        <div style='font-size:1.2rem; font-weight:600; color:#6366f1; margin-top:1rem;'>
            Chat parsed successfully!
        </div>
        <div style='margin-top:0.5rem;'>Select a user and click <strong>Analyse Now</strong> to begin.</div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Analyse Now", key="main_analyse_btn", type="primary", use_container_width=True):
            st.session_state.analyzed = True
            st.rerun()
            
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ── TABS ──────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Activity", "Words & Emojis",
    "ML Insights", "Summary"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><span class=material-symbols-outlined>bar_chart</span> Chat Overview</div>', unsafe_allow_html=True)

    num_messages, words, num_media, links = helper.fetch_stats(selected_user, df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", f"{num_messages:,}")
    c2.metric("Total Words",    f"{words:,}")
    c3.metric("Media Shared",   f"{num_media:,}")
    c4.metric("Links Shared",   f"{links:,}")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Most Active Users (group only)
    if selected_user == 'Overall' and df['user'].nunique() > 1:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>emoji_events</span> Most Active Users</div>', unsafe_allow_html=True)
        col_bar, col_pie = st.columns([3, 2])

        counts, percent_df = helper.most_busy_users(df)
        top_n = min(10, len(counts))

        with col_bar:
            fig, ax = _fig(7, 4)
            bars = ax.barh(
                counts.index[:top_n][::-1],
                counts.values[:top_n][::-1],
                color=GRAD_COLORS[:top_n][::-1],
                edgecolor='none', height=0.6
            )
            for bar, val in zip(bars, counts.values[:top_n][::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:,}', va='center', fontsize=9, color='#a5b4fc')
            ax.set_xlabel('Messages', color='#a5b4fc')
            ax.set_title('Messages per User', color='#c4c9e8', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_pie:
            fig_pie = px.pie(
                percent_df.head(top_n),
                values='percent', names='name',
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                font_color='#cbd5e1',
                legend=dict(font=dict(size=10)),
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
            )
            fig_pie.update_traces(textfont_size=11, textposition='inside')
            st.plotly_chart(fig_pie, use_container_width=True)

    # Date range info
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("First Message", str(df['date'].min().date()))
    col_b.metric("Last Message",  str(df['date'].max().date()))
    days = (df['date'].max() - df['date'].min()).days + 1
    col_c.metric("Chat Duration", f"{days:,} days")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><span class=material-symbols-outlined>language</span> Activity Analysis</div>', unsafe_allow_html=True)

    col_tl1, col_tl2 = st.columns(2)
    with col_tl1:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>calendar_today</span> Monthly Timeline</div>', unsafe_allow_html=True)
        timeline = helper.monthly_timeline(selected_user, df)
        if not timeline.empty:
            fig_mt = px.area(
                timeline, x='time', y='message',
                color_discrete_sequence=['#6366f1'],
                labels={'message': 'Messages', 'time': 'Month'},
            )
            fig_mt.update_traces(
                line_color='#a855f7', fillcolor='rgba(99,102,241,0.2)',
                hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>',
            )
            fig_mt.update_layout(
                paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#cbd5e1',
                xaxis_tickangle=-45, margin=dict(t=20, b=40, l=40, r=20), height=300
            )
            st.plotly_chart(fig_mt, use_container_width=True)

    with col_tl2:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>calendar_month</span> Daily Timeline</div>', unsafe_allow_html=True)
        daily = helper.daily_timeline(selected_user, df)
        if not daily.empty:
            fig_dt = px.line(
                daily, x='only_date', y='message',
                color_discrete_sequence=['#ec4899'],
                labels={'message': 'Messages', 'only_date': 'Date'},
            )
            fig_dt.update_traces(hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>')
            fig_dt.update_layout(
                paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#cbd5e1',
                margin=dict(t=20, b=40, l=40, r=20), height=300
            )
            st.plotly_chart(fig_dt, use_container_width=True)

    # Day / Month bars
    col_d, col_m = st.columns(2)
    with col_d:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>calendar_today</span> Busiest Days</div>', unsafe_allow_html=True)
        day_map = helper.week_activity_map(selected_user, df)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_map = day_map.reindex([d for d in day_order if d in day_map.index])
        fig, ax = _fig(4, 3)
        colors = [GRAD_COLORS[i % len(GRAD_COLORS)] for i in range(len(day_map))]
        ax.bar(day_map.index, day_map.values, color=colors, edgecolor='none')
        ax.set_xlabel('Day', color='#a5b4fc')
        ax.set_ylabel('Messages', color='#a5b4fc')
        plt.xticks(rotation=30, ha='right')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_m:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>calendar_today</span> Busiest Months</div>', unsafe_allow_html=True)
        mon_map = helper.month_activity_map(selected_user, df)
        fig, ax = _fig(4, 3)
        ax.bar(mon_map.index, mon_map.values,
               color=GRAD_COLORS[:len(mon_map)], edgecolor='none')
        ax.set_xlabel('Month', color='#a5b4fc')
        ax.set_ylabel('Messages', color='#a5b4fc')
        plt.xticks(rotation=30, ha='right')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Heatmap
    st.markdown('<div class="sub-header"><span class=material-symbols-outlined>local_fire_department</span> Activity Heatmap (Hour × Day)</div>', unsafe_allow_html=True)
    heatmap_data = helper.activity_heatmap(selected_user, df)
    if not heatmap_data.empty:
        fig, ax = _fig(10, 3)
        sns.heatmap(
            heatmap_data, ax=ax,
            cmap='plasma', linewidths=0.5,
            linecolor='#0d0d1a', cbar_kws={'shrink': 0.7},
            fmt='.0f', annot=False,
        )
        ax.set_title('Message Frequency by Hour & Day', color='#c4c9e8', fontweight='bold')
        ax.set_xlabel('Hour Period', color='#a5b4fc')
        ax.set_ylabel('')
        plt.xticks(rotation=60, ha='right', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WORDS & EMOJIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><span class=material-symbols-outlined>sort_by_alpha</span> Words & Emojis</div>', unsafe_allow_html=True)

    # Columns for Words and Wordcloud
    col_w1, col_w2 = st.columns([1, 1.5])
    
    with col_w1:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>menu_book</span> Top Words</div>', unsafe_allow_html=True)
        word_df = helper.most_common_words(selected_user, df)
        if not word_df.empty:
            fig, ax = _fig(4, 4.5)
            # Only top 15 words to save space
            top_w = word_df.head(15)
            bars = ax.barh(top_w['Word'][::-1], top_w['Count'][::-1],
                           color=GRAD_COLORS[:len(top_w)][::-1], edgecolor='none')
            ax.set_xlabel('Count', color='#a5b4fc')
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_w2:
        st.markdown('<div class="sub-header"><span class=material-symbols-outlined>cloud</span> Word Cloud</div>', unsafe_allow_html=True)
        with st.spinner("Generating word cloud…"):
            wc = helper.create_wordcloud(selected_user, df)
            fig_wc, ax_wc = plt.subplots(figsize=(6, 4.5))
            fig_wc.patch.set_facecolor('#0d0d1a')
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close()

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    
    # Emoji analysis (full width but smaller height pie or side-by-side)
    st.markdown('<div class="sub-header"><span class=material-symbols-outlined>sentiment_satisfied</span> Top Emojis</div>', unsafe_allow_html=True)
    emoji_df = helper.emoji_helper(selected_user, df)
    
    col_e1, col_e2 = st.columns([1, 1])
    
    if not emoji_df.empty:
        with col_e1:
            fig_em = px.pie(
                emoji_df, values='Count', names='Emoji',
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.35,
            )
            fig_em.update_layout(
                paper_bgcolor='#0f172a', font_color='#cbd5e1',
                margin=dict(t=20, b=20, l=20, r=20), height=450,
                legend=dict(font=dict(size=22), itemsizing='constant')
            )
            st.plotly_chart(fig_em, use_container_width=True)
            
        with col_e2:
            st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
            
            # Custom Centered HTML Table
            html_table = f"""
            <div style="max-height: 400px; overflow-y: auto; border: 1px solid #334155; border-radius: 8px; background-color:#0f172a;">
            <table style="width:100%; border-collapse: collapse; text-align:center; color:#cbd5e1; font-size:0.95rem;">
                <thead style="position: sticky; top: 0; background-color: #1e293b; z-index: 1; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    <tr>
                        <th style="padding:12px; border-bottom: 2px solid #334155; font-weight:600; color:#f8fafc;">Emoji</th>
                        <th style="padding:12px; border-bottom: 2px solid #334155; font-weight:600; color:#f8fafc;">Count</th>
                    </tr>
                </thead>
                <tbody>
            """
            for _, row in emoji_df.iterrows():
                html_table += f"""
                <tr style='border-bottom: 1px solid #334155;'>
                    <td style='padding:10px; font-size:1.2rem;'>{row['Emoji']}</td>
                    <td style='padding:10px; font-weight:600;'>{row['Count']}</td>
                </tr>
                """
            html_table += "</tbody></table></div>"
            st.html(html_table)
    else:
        st.info("No emojis found in the selected messages.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML INSIGHTS  ⭐ KEY RESUME FEATURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="section-header"><span class=material-symbols-outlined>smart_toy</span> ML Insights <span class="ml-badge">AI POWERED</span></div>',
        unsafe_allow_html=True
    )

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "Sentiment", "Topics", "Behavior", "Spam"
    ])

    # ── ML TAB 1: SENTIMENT ───────────────────────────────────────────────────
    with ml_tab1:
        st.markdown('<div class="sub-header">Sentiment Analysis <span class="ml-badge">VADER + TextBlob Ensemble</span></div>', unsafe_allow_html=True)
        with st.spinner("Running sentiment analysis…"):
            sent_results = ml_models.analyze_sentiment(df, selected_user)

        label = sent_results['overall_label']
        score = sent_results['overall_score']
        score_color = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#eab308'}[label]
        emoji_map   = {'Positive': '<span class=material-symbols-outlined>sentiment_satisfied</span>', 'Negative': '<span class=material-symbols-outlined>sentiment_dissatisfied</span>', 'Neutral': '<span class=material-symbols-outlined>sentiment_neutral</span>'}

        # Overall card
        st.markdown(f"""
        <div class='sentiment-card sentiment-{label.lower()}' style='max-width:400px; margin:0 auto 1.5rem;'>
            <div style='font-size:3rem;'>{emoji_map[label]}</div>
            <div style='font-size:1.8rem; font-weight:800; color:{score_color};'>{label}</div>
            <div style='font-size:0.9rem; color:#cbd5e1; margin-top:4px;'>
                Overall Score: <strong style='color:{score_color};'>{score:+.4f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        per_user_df = sent_results.get('per_user', pd.DataFrame())
        if not per_user_df.empty:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_s = px.bar(
                    per_user_df.sort_values('avg_score', ascending=True),
                    x='avg_score', y='user', orientation='h',
                    color='avg_score', color_continuous_scale='RdYlGn',
                    labels={'avg_score': 'Avg Score', 'user': 'User'},
                    title='Average Score',
                )
                fig_s.update_layout(
                    paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                    font_color='#cbd5e1', coloraxis_showscale=False,
                    title_font_color='#f8fafc', height=300,
                    margin=dict(t=40, b=20, l=10, r=20),
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with col_s2:
                fig_stack = go.Figure()
                for sentiment, color in [('positive', '#22c55e'), ('neutral', '#eab308'), ('negative', '#ef4444')]:
                    fig_stack.add_trace(go.Bar(
                        y=per_user_df['user'], x=per_user_df[sentiment],
                        name=sentiment.capitalize(), orientation='h', marker_color=color,
                    ))
                fig_stack.update_layout(
                    barmode='stack', paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                    font_color='#cbd5e1', title='Sentiment Distribution',
                    title_font_color='#f8fafc', height=300,
                    margin=dict(t=40, b=20, l=10, r=20),
                )
                st.plotly_chart(fig_stack, use_container_width=True)

        # Sample messages
        per_msg_df = sent_results.get('per_message', pd.DataFrame())
        if not per_msg_df.empty:
            with st.expander("View Sample Sentiment-Scored Messages"):
                st.dataframe(
                    per_msg_df[['user', 'message', 'sentiment_score', 'sentiment_label']].head(50),
                    use_container_width=True, hide_index=True
                )

    # ── ML TAB 2: TOPICS ──────────────────────────────────────────────────────
    with ml_tab2:
        st.markdown('<div class="sub-header">Topic Modeling <span class="ml-badge">LDA</span></div>', unsafe_allow_html=True)
        n_topics = st.slider("Number of Topics", 2, 8, 5, key='lda_slider')
        with st.spinner("Running LDA topic modeling…"):
            topics = ml_models.run_topic_modeling(df, n_topics=n_topics)

        if topics:
            for t in topics:
                words_html = ' '.join(
                    f'<span class="pill pill-purple">{w}</span>' for w in t['words']
                )
                st.markdown(f"""
                <div class="topic-card">
                    <span class="topic-number">{t['topic_id']}</span>
                    <strong style='color:#c4c9e8;'>Topic {t['topic_id']}</strong>
                    <div style='margin-top:8px;'>{words_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough data for topic modeling.")

    # ── ML TAB 3: BEHAVIOR ────────────────────────────────────────────────────
    with ml_tab3:
        st.markdown('<div class="sub-header">User Behavior Prediction <span class="ml-badge">Random Forest</span></div>', unsafe_allow_html=True)
        with st.spinner("Training behavior model…"):
            behavior = ml_models.predict_peak_hours(df)

        if behavior:
            bucket_emoji = {'Morning': '<span class=material-symbols-outlined>wb_twilight</span>', 'Afternoon': '<span class=material-symbols-outlined>light_mode</span>', 'Evening': '<span class=material-symbols-outlined>location_city</span>', 'Night': '<span class=material-symbols-outlined>dark_mode</span>'}
            cols = st.columns(min(3, len(behavior)))
            for i, (user, info) in enumerate(behavior.items()):
                with cols[i % len(cols)]:
                    acc_text = f"RF Accuracy: **{info['model_accuracy']}%**" if info['model_accuracy'] else ""
                    st.markdown(f"""
                    <div class='topic-card'>
                        <div style='font-size:1.5rem;'>{bucket_emoji.get(info['bucket'],'<span class=material-symbols-outlined>schedule</span>')}</div>
                        <div style='font-weight:700; color:#c4c9e8; margin:6px 0;'>{user}</div>
                        <div><span class='pill pill-blue'>Peak Hour: {info['peak_hour']:02d}:00</span></div>
                        <div style='margin-top:6px;'><span class='pill pill-purple'>{info['peak_day']}</span></div>
                        <div style='margin-top:6px;'><span class='pill pill-green'>{info['bucket']}</span></div>
                        <div style='font-size:0.75rem; color:#cbd5e1; margin-top:8px;'>{acc_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Hour activity chart
            st.markdown('<div class="sub-header"><span class=material-symbols-outlined>trending_up</span> Hourly Activity by User</div>', unsafe_allow_html=True)
            fig_h = go.Figure()
            for i, (user, info) in enumerate(behavior.items()):
                hs = info['hourly_series']
                fig_h.add_trace(go.Scatter(
                    x=hs.index, y=hs.values,
                    name=user, mode='lines+markers',
                    line=dict(color=GRAD_COLORS[i % len(GRAD_COLORS)], width=2),
                    marker=dict(size=5),
                    hovertemplate=f'<b>{user}</b><br>Hour: %{{x}}:00<br>Messages: %{{y}}<extra></extra>',
                ))
            fig_h.update_layout(
                paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                font_color='#cbd5e1', title='Messages by Hour of Day',
                title_font_color='#f8fafc',
                xaxis=dict(title='Hour', dtick=2),
                yaxis=dict(title='Message Count'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=40, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Not enough data to model behavior.")

    # ── ML TAB 4: SPAM ────────────────────────────────────────────────────────
    with ml_tab4:
        st.markdown('<div class="sub-header">Spam & Abusive Message Detection <span class="ml-badge">TF-IDF + Logistic Regression</span></div>', unsafe_allow_html=True)
        with st.spinner("Running spam detection…"):
            spam_df = ml_models.detect_spam(df)

        if not spam_df.empty:
            label_counts = spam_df['spam_label'].value_counts()
            total = len(spam_df)

            cs = st.columns(3)
            for col, (lbl, cnt) in zip(cs, label_counts.items()):
                color_map = {'Normal': '#22c55e', 'Spam': '#f97316', 'Abusive': '#ef4444'}
                col.metric(f"{lbl} Messages", f"{cnt:,}",
                           delta=f"{cnt/total*100:.1f}%")

            # Distribution chart
            fig_sp = px.pie(
                values=label_counts.values, names=label_counts.index,
                color_discrete_map={'Normal': '#22c55e', 'Spam': '#f97316', 'Abusive': '#ef4444'},
                hole=0.4,
            )
            fig_sp.update_layout(
                paper_bgcolor='#0f172a', font_color='#cbd5e1',
                margin=dict(t=20, b=20, l=20, r=20), height=300,
            )
            st.plotly_chart(fig_sp, use_container_width=True)

            # Flagged messages
            flagged = spam_df[spam_df['spam_label'] != 'Normal']
            if not flagged.empty:
                st.markdown(f'<div class="sub-header"><span class=material-symbols-outlined>warning</span> Flagged Messages ({len(flagged)} found)</div>', unsafe_allow_html=True)
                st.dataframe(
                    flagged[['user', 'message', 'spam_label', 'spam_confidence']].head(50),
                    use_container_width=True, hide_index=True
                )
            else:
                st.success("No spam or abusive messages detected!")
        else:
            st.info("No processable messages found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="section-header"><span class=material-symbols-outlined>edit_document</span> AI Chat Summary <span class="ml-badge">NLP</span></div>',
        unsafe_allow_html=True
    )
    n_sent = st.slider("Number of summary sentences", 3, 10, 5, key='summ_slider')
    with st.spinner("Generating NLP summary…"):
        summary = ml_models.summarize_chat(df, selected_user, n_sentences=n_sent)

    st.markdown(f"""
    <div style='background-color:#1e293b,rgba(168,85,247,0.06));
        border:1px solid #334155; border-radius:18px; padding:2rem;
        line-height:1.9; font-size:1.05rem; color:#cbd5e1; margin:1rem 0;'>
        <div style='margin-bottom:1rem; color:#a5b4fc; font-size:0.8rem; font-weight:700; letter-spacing:0.1em;'>
        <span class=material-symbols-outlined>edit_document</span> AUTO-GENERATED SUMMARY
        </div>
        {summary}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"><span class=material-symbols-outlined>lightbulb</span> Quick Insights</div>', unsafe_allow_html=True)

    # Quick facts
    num_messages, words, num_media, _ = helper.fetch_stats(selected_user, df)
    avg_per_day = num_messages / max(1, (df['date'].max() - df['date'].min()).days)
    most_active_user = df['user'].value_counts().idxmax()
    most_active_hour = df['hour'].value_counts().idxmax()

    insights = [
        (f"{num_messages:,} messages exchanged", "pill-blue"),
        (f"~{int(words/max(num_messages,1))} words per message", "pill-purple"),
        (f"{num_media:,} media files shared", "pill-green"),
        (f"~{avg_per_day:.1f} messages per day on average", "pill-yellow"),
        (f"Most active: {most_active_user}", "pill-purple"),
        (f"Peak hour: {most_active_hour:02d}:00", "pill-blue"),
    ]
    pills_html = ' '.join(f'<span class="pill {cls}">{text}</span>' for text, cls in insights)
    st.markdown(f"<div style='margin:1rem 0;'>{pills_html}</div>", unsafe_allow_html=True)
