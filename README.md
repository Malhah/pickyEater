
#  Food Recommender CLI

A personalized food and restaurant recommendation system using logistic regression and user history.
You rate dishes you've eaten, and the app learns your preferences to recommend where — and what — you should eat next.

---

##  Features

-  Add and view restaurants with price and cuisine
-  Rate individual dishes with personal score and "Liked" flag
-  Recommender system predicts which restaurants you'll like
-  Suggests up to 3 dishes you've enjoyed in similar cuisines
-  Confidence scoring for every recommendation
-  Datasets saved locally as CSV for full transparency and export

---

##  How It Works

1. You add restaurants to a local CSV file.
2. You rate dishes you’ve tried, including:
   - Dish name
   - Personal rating (1–10)
   - If you liked it (binary)
   - Date visited
3. The model trains on your rating history using `LogisticRegression`.
4. You run recommendations — the top restaurant and matching dishes are shown.

---

##  Example Output

```
===  Food Recommender Menu ===
3. Get food recommendations

🍴 Recommendation Summary:
🔹 Where to eat:
   → Thai Max (Thai) - Confidence: 0.92
🔹 What to eat:
   • 'Pad Thai' (Rated 9/10)
   • 'Green Curry' (Rated 8/10)
```

---

##  File Structure

| File                 | Purpose                             |
|----------------------|-------------------------------------|
| `food_recommender_cli.py` | Main CLI logic (add, rate, recommend) |
| `restaurants.csv`    | Local dataset of restaurants        |
| `user_data.csv`      | History of dish ratings             |
| `recommendations.csv`| Model predictions on restaurants    |

---

## 🧠 Tech Stack

- Python 3.8+
- scikit-learn (`LogisticRegression`, `OneHotEncoder`)
- pandas
- CLI-based user interface

---

## 🛠 Future Additions

- [ ] Streamlit web version with UI
- [ ] Data visualizations (top cuisines, trends)
- [ ] More models (e.g., Random Forest or XGBoost)
- [ ] Collaborative filtering for shared preferences

---


