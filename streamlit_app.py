import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import os
from datetime import datetime

def normalize_cuisine(cuisine):
    cuisine = cuisine.lower()
    if "burger" in cuisine:
        return "Burger"
    if "sushi" in cuisine:
        return "Sushi"
    if "italian" in cuisine or "pizza" in cuisine:
        return "Italian"
    if "thai" in cuisine:
        return "Thai"
    if "kebab" in cuisine:
        return "Middle Eastern"
    return cuisine.title()

class FoodRecommender:
    def __init__(self):
        self.model = LogisticRegression()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.trained = False

    def fit(self, user_data: pd.DataFrame):
        user_data = user_data.copy()
        user_data["Cuisine"] = user_data["Cuisine"].apply(normalize_cuisine)
        user_data = user_data[["Cuisine", "Rating", "Price", "Liked"]]
        encoded = self.encoder.fit_transform(user_data[["Cuisine"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        features = pd.concat([encoded_df.reset_index(drop=True), user_data[["Rating", "Price"]].reset_index(drop=True)], axis=1)
        labels = user_data["Liked"]
        self.model.fit(features, labels)
        self.trained = True

    def predict(self, new_places: pd.DataFrame):
        new_places = new_places.copy()
        new_places["Cuisine"] = new_places["Cuisine"].apply(normalize_cuisine)
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        encoded = self.encoder.transform(new_places[["Cuisine"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        X_new = pd.concat([encoded_df.reset_index(drop=True), new_places[["Rating", "Price"]].reset_index(drop=True)], axis=1)
        new_places["Predicted_Like"] = self.model.predict(X_new)
        new_places["Confidence"] = self.model.predict_proba(X_new)[:, 1]
        return new_places.sort_values("Confidence", ascending=False)

st.title("Food Recommender")

if not os.path.exists("user_data.csv"):
    st.warning("No user history found.")
if not os.path.exists("restaurants.csv"):
    st.warning("No restaurants found.")

if os.path.exists("restaurants.csv"):
    restaurants_df = pd.read_csv("restaurants.csv")
else:
    restaurants_df = pd.DataFrame(columns=["Name", "Cuisine", "Rating", "Price"])

if os.path.exists("user_data.csv"):
    user_data_df = pd.read_csv("user_data.csv")
else:
    user_data_df = pd.DataFrame(columns=["Name", "Cuisine", "Rating", "Price", "Liked", "DateVisited", "Dish", "PersonalRating"])

tab1, tab2, tab3 = st.tabs(["Add Restaurant", "Rate Dish", "Get Recommendation"])

# Tab 1: Add restaurant
with tab1:
    st.header("Add New Restaurant")
    with st.form("add_restaurant"):
        name = st.text_input("Restaurant Name")
        cuisine = st.text_input("Cuisine")
        rating = st.slider("Average Rating", 0.0, 5.0, 4.0, 0.1)
        price = st.selectbox("Price Level", [1, 2, 3, 4])
        submitted = st.form_submit_button("Add Restaurant")
        if submitted:
            new_row = pd.DataFrame([{
                "Name": name,
                "Cuisine": normalize_cuisine(cuisine),
                "Rating": rating,
                "Price": price
            }])
            restaurants_df = pd.concat([restaurants_df, new_row], ignore_index=True)
            restaurants_df.to_csv("restaurants.csv", index=False)
            st.success(f"{name} added successfully!")

# Tab 2: Rate a dish
with tab2:
    st.header("Rate a Dish You've Tried")
    if not restaurants_df.empty:
        selected = st.selectbox("Select Restaurant", restaurants_df["Name"])
        selected_row = restaurants_df[restaurants_df["Name"] == selected].iloc[0]
        with st.form("rate_dish"):
            dish = st.text_input("Dish Name")
            personal_rating = st.slider("Dish Rating", 1, 10, 7)
            liked = st.radio("Did you like it?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            date = st.date_input("Date Visited", value=datetime.today())
            submit_rating = st.form_submit_button("Save Rating")
            if submit_rating:
                new_entry = {
                    "Name": selected_row["Name"],
                    "Cuisine": normalize_cuisine(selected_row["Cuisine"]),
                    "Rating": selected_row["Rating"],
                    "Price": selected_row["Price"],
                    "Liked": liked,
                    "DateVisited": date.strftime('%Y-%m-%d'),
                    "Dish": dish,
                    "PersonalRating": personal_rating
                }
                user_data_df = pd.concat([user_data_df, pd.DataFrame([new_entry])], ignore_index=True)
                user_data_df.to_csv("user_data.csv", index=False)
                st.success("Rating saved!")

# Tab 3: Get recommendation
with tab3:
    st.header("Your Personalized Recommendation")
    if len(user_data_df) >= 3 and len(restaurants_df) >= 1:
        model = FoodRecommender()
        model.fit(user_data_df)
        result = model.predict(restaurants_df)
        top_restaurant = result.iloc[0]
        cuisine = normalize_cuisine(top_restaurant["Cuisine"])

        dishes = user_data_df[
            (user_data_df["Cuisine"].apply(normalize_cuisine) == cuisine) &
            (user_data_df["PersonalRating"] >= 8)
        ].sort_values("PersonalRating", ascending=False).head(3)

        st.subheader("Where to eat:")
        st.markdown(f"**{top_restaurant['Name']}** ({cuisine}) – Confidence: **{top_restaurant['Confidence']:.2f}**")

        st.subheader("What to eat:")
        if not dishes.empty:
            for _, row in dishes.iterrows():
                st.markdown(f"• **{row['Dish']}** (Rated {row['PersonalRating']}/10)")
        else:
            st.info("No matching dishes found. Try something new!")
    else:
        st.warning("Add at least 3 dish ratings and 1 restaurant to get recommendations.")
