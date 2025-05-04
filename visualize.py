
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("user_data.csv")

# Convert DateVisited to datetime
df["DateVisited"] = pd.to_datetime(df["DateVisited"], errors='coerce')

# --- 1. Average rating by cuisine ---
plt.figure(figsize=(8, 5))
avg_rating = df.groupby("Cuisine")["PersonalRating"].mean().sort_values(ascending=False)
avg_rating.plot(kind="bar", title=" Average Dish Rating by Cuisine", ylabel="Rating (1–10)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 2. Count of liked dishes per cuisine ---
plt.figure(figsize=(8, 5))
liked = df[df["Liked"] == 1]
liked_count = liked["Cuisine"].value_counts()
liked_count.plot(kind="bar", title=" Liked Dishes per Cuisine", ylabel="Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3. Rating trend over time ---
plt.figure(figsize=(10, 5))
df.sort_values("DateVisited").plot(x="DateVisited", y="PersonalRating", kind="line", title=" Personal Ratings Over Time")
plt.ylabel("Rating (1–10)")
plt.grid(True)
plt.tight_layout()
plt.show()
