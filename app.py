import streamlit as st
import praw
import pandas as pd
import joblib
from datetime import datetime, timezone
from collections import defaultdict
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()
# =========================
# CONFIG
# =========================
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

assert CLIENT_ID, "REDDIT_CLIENT_ID not loaded"
assert CLIENT_SECRET, "REDDIT_CLIENT_SECRET not loaded"
assert USER_AGENT, "REDDIT_USER_AGENT not loaded"

MODEL_PATH = "spread_risk_model.joblib"
# =========================
# Reddit + Model
# =========================
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)


pipeline = joblib.load(MODEL_PATH)

# =========================
# Helpers
# =========================
def build_tree(reactions):
    children = defaultdict(list)
    for r in reactions:
        if r["parent"]:
            children[r["parent"]].append(r["id"])
    return children

def cascade_depth(node, children, depth=0):
    if node not in children:
        return depth
    return max(
        cascade_depth(child, children, depth + 1)
        for child in children[node]
    )

def cascade_width(children):
    return max((len(v) for v in children.values()), default=0)

def early_reactions(reactions, source_time, minutes):
    cutoff = source_time.timestamp() + minutes * 60
    return sum(1 for r in reactions if r["time"].timestamp() <= cutoff)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Early Spread Risk Dashboard",
    layout="wide"
)

st.title("Early Spread Risk Analyzer")
st.caption("Predicts early diffusion risk — not misinformation")

url = st.text_input("🔗 Paste a Reddit thread URL")

if url:
    with st.spinner("Analyzing Reddit thread..."):
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)

        created_time = datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        )

        reactions = []
        for c in submission.comments.list():
            reactions.append({
                "id": c.id,
                "parent": c.parent_id.replace("t1_", "").replace("t3_", ""),
                "time": datetime.fromtimestamp(c.created_utc, tz=timezone.utc)
            })

        children = build_tree(reactions)

        features = {
            "total_reactions": len(reactions),
            "reactions_30min": early_reactions(reactions, created_time, 30),
            "reactions_60min": early_reactions(reactions, created_time, 60),
            "cascade_depth": cascade_depth(submission.id, children),
            "cascade_width": cascade_width(children),
            "tweets_per_day": 0,
            "avg_gap_minutes": 0,
            "burstiness": 0,
            "topic_entropy": 0,
        }

        X = pd.DataFrame([features])
        risk_prob = pipeline.predict_proba(X)[0][1]

    # =========================
    # Reddit preview card
    # =========================
    with st.container(border=True):
        cols = st.columns([3, 1])

        with cols[0]:
            st.subheader(submission.title)
            st.caption(
                f"r/{submission.subreddit} • "
                f"{submission.score} upvotes • "
                f"{submission.num_comments} comments"
            )

        with cols[1]:
            if submission.thumbnail and submission.thumbnail.startswith("http"):
                st.image(submission.thumbnail, use_container_width=True)

    st.markdown("---")

    # =========================
    # Risk + Features
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Spread Risk")
            st.metric(
                "Risk Probability",
                f"{risk_prob:.3f}",
                delta="HIGH" if risk_prob >= 0.5 else "LOW",
            )

            st.write("**Interpretation:**")
            if risk_prob >= 0.5:
                st.warning("Rapid early growth detected. High diffusion risk.")
            else:
                st.success("Growth appears organic and slow.")

    with col2:
        with st.container(border=True):
            st.subheader("Model Inputs")
            st.dataframe(X.T, use_container_width=True)

    # =========================
    # Engagement plot
    # =========================
    with st.container(border=True):
        st.subheader("Comment Growth Over Time")

        times = sorted([r["time"] for r in reactions])
        counts = list(range(1, len(times) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, counts, linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Comments")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # =========================
    # Cascade stats
    # =========================
    with st.container(border=True):
        st.subheader("Cascade Structure")
        st.write(f"**Depth:** {features['cascade_depth']}")
        st.write(f"**Max Width:** {features['cascade_width']}")
