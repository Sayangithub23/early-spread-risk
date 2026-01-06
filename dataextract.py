# =========================
# Imports
# =========================
import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
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
# Thread Loader
# =========================
def load_thread(thread_path):
    thread_id = os.path.basename(thread_path)

    source_path = os.path.join(
        thread_path, "source-tweet", f"{thread_id}.json"
    )
    source = load_json(source_path)

    source_id = source["id_str"]
    source_time = parse_time(source["created_at"])

    reactions = []
    reactions_path = os.path.join(thread_path, "reactions")

    for file in os.listdir(reactions_path):
        r = load_json(os.path.join(reactions_path, file))
        reactions.append({
            "id": r["id_str"],
            "parent": r["in_reply_to_status_id_str"],
            "time": parse_time(r["created_at"])
        })

    return source_id, source_time, reactions


# =========================
# Cascade utilities
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

def early_reactions(source_time, reactions, minutes):
    cutoff = source_time.timestamp() + minutes * 60
    return sum(1 for r in reactions if r["time"].timestamp() <= cutoff)


# =========================
# Dataset Path
# =========================
BASE_PATH = r"F:\PHEME\pheme-rnr-dataset"


# =========================
# Main Extraction Loop
# =========================
rows = []

events = [
    e for e in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, e))
]

for event in tqdm(events, desc="Events"):
    event_path = os.path.join(BASE_PATH, event)

    for label in ["rumours", "non-rumours"]:
        label_path = os.path.join(event_path, label)
        if not os.path.exists(label_path):
            continue

        threads = os.listdir(label_path)

        for thread_id in tqdm(
            threads,
            desc=f"{event}/{label}",
            leave=False
        ):
            thread_path = os.path.join(label_path, thread_id)

            try:
                source_id, source_time, reactions = load_thread(thread_path)
                children = build_tree(reactions)

                rows.append({
                    "event": event,
                    "thread_id": thread_id,
                    "source_type": label,  # metadata only
                    "total_reactions": len(reactions),
                    "reactions_30min": early_reactions(source_time, reactions, 30),
                    "reactions_60min": early_reactions(source_time, reactions, 60),
                    "cascade_depth": cascade_depth(source_id, children),
                    "cascade_width": cascade_width(children),
                })

            except Exception as e:
                print(f"Skipping {thread_id}: {e}")


# =========================
# Create DataFrame
# =========================
df = pd.DataFrame(rows)

print("\nTotal threads processed:", len(df))
print(df.head())


# =========================
# CREATE OUR OWN LABEL
# =========================
def assign_spread_risk(row):
    if row["reactions_60min"] >= 15 or row["cascade_depth"] >= 8:
        return 1
    return 0

df["spread_risk"] = df.apply(assign_spread_risk, axis=1)


# =========================
# Save Final Dataset
# =========================
output_path = r"F:\PHEME\pheme_spread_risk_dataset.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved final dataset to: {output_path}")
print(df["spread_risk"].value_counts())
