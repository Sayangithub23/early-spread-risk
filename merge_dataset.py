import pandas as pd

#Load datasets
df_diff = pd.read_csv(r"F:\PHEME\pheme_spread_risk_dataset.csv")
df_user = pd.read_csv(r"F:\PHEME\pheme_user_behavior.csv")

print("Diffusion threads:", len(df_diff))
print("User behavior rows:", len(df_user))

# Merge (LEFT JOIN)
df = df_diff.merge(df_user, on="thread_id", how="left")

# Fill missing user features
user_cols = ["tweets_per_day", "avg_gap_minutes", "burstiness", "topic_entropy"]

for col in user_cols:
    df[col] = df[col].fillna(0)

print("\nMerged dataset shape:", df.shape)
print(df.head())

# Save final dataset
output_path = r"F:\PHEME\pheme_final_dataset.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved FINAL dataset to: {output_path}")
user_cols = ["tweets_per_day", "avg_gap_minutes", "burstiness", "topic_entropy"]

print(df[user_cols].describe())
