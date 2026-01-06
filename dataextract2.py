# =========================
# Imports
# =========================
import os
import json
import pandas as pd
from datetime import datetime
import math
from collections import Counter
from tqdm import tqdm


# =========================
# Helpers
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_time(t):
    return datetime.strptime(t, "%a %b %d %H:%M:%S %z %Y")


# =========================
# Topic entropy
# =========================
def topic_entropy(texts):
    words = []
    for t in texts:
        words.extend(t.lower().split())

    if not words:
        return 0.0

    counts = Counter(words)
    total = sum(counts.values())

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-9)

    return entropy


# =========================
# User behavior extraction
# =========================
def extract_user_features(thread_path):
    times = []
    texts = []

    for file in os.listdir(thread_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(thread_path, file)

        try:
            # Try normal JSON
            with open(file_path, "r", encoding="utf-8") as f:
                tweet = json.load(f)

            if "created_at" in tweet:
                times.append(parse_time(tweet["created_at"]))
                texts.append(tweet.get("text", ""))

        except Exception:
            # Fallback: multiple JSON objects in one file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tweet = json.loads(line)
                        if "created_at" in tweet:
                            times.append(parse_time(tweet["created_at"]))
                            texts.append(tweet.get("text", ""))
            except Exception:
                continue  # truly broken file

    if len(times) < 2:
        return None

    times.sort()

    days = (times[-1] - times[0]).days + 1
    tweets_per_day = len(times) / max(days, 1)

    gaps = [
        (times[i] - times[i - 1]).total_seconds() / 60
        for i in range(1, len(times))
    ]
    avg_gap = sum(gaps) / len(gaps)

    burst = 1
    for i in range(len(times)):
        count = 1
        for j in range(i + 1, len(times)):
            if (times[j] - times[i]).total_seconds() <= 3600:
                count += 1
            else:
                break
        burst = max(burst, count)

    return {
        "tweets_per_day": tweets_per_day,
        "avg_gap_minutes": avg_gap,
        "burstiness": burst,
        "topic_entropy": topic_entropy(texts)
    }

# =========================
# Dataset path
# =========================
EXTENDED_BASE = r"F:\PHEME\Extended-Pheme-Dataset-master\en"  


# =========================
# Main loop
# =========================
rows = []

events = [
    e for e in os.listdir(EXTENDED_BASE)
    if os.path.isdir(os.path.join(EXTENDED_BASE, e))
]

for event in tqdm(events, desc="Events"):
    event_path = os.path.join(EXTENDED_BASE, event)

    thread_ids = [
        t for t in os.listdir(event_path)
        if os.path.isdir(os.path.join(event_path, t))
    ]

    for thread_id in thread_ids:
        thread_path = os.path.join(event_path, thread_id)

        try:
            features = extract_user_features(thread_path)
            if features is None:
                continue

            rows.append({
                "thread_id": thread_id,
                **features
            })

        except Exception as e:
            print(f"Skipping {thread_id}: {e}")

# =========================
# Create DataFrame
# =========================
df_user = pd.DataFrame(rows)
print(df_user.head())
print("Users processed:", len(df_user))


# =========================
# Save
# =========================
output_path = r"F:\PHEME\pheme_user_behavior.csv"
df_user.to_csv(output_path, index=False)
print(f"Saved user features to: {output_path}")
