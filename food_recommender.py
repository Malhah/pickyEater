import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import os
from datetime import datetime

class FoodRecommender:
    def __init__(self):
        self.model = LogisticRegression()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.trained = False

    def fit(self, user_data: pd.DataFrame):
        user_data = user_data[["Cuisine", "Rating", "Price", "Liked"]]
        encoded = self.encoder.fit_transform(user_data[["Cuisine"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        features = pd.concat([encoded_df.reset_index(drop=True), user_data[["Rating", "Price"]].reset_index(drop=True)], axis=1)
        labels = user_data["Liked"]
        self.model.fit(features, labels)
        self.trained = True

    def predict(self, new_places: pd.DataFrame):
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        encoded = self.encoder.transform(new_places[["Cuisine"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        X_new = pd.concat([encoded_df.reset_index(drop=True), new_places[["Rating", "Price"]].reset_index(drop=True)], axis=1)

        new_places = new_places.copy()
        new_places["Predicted_Like"] = self.model.predict(X_new)
        new_places["Confidence"] = self.model.predict_proba(X_new)[:, 1]
        return new_places.sort_values("Confidence", ascending=False)

def add_restaurant_interactively(file="restaurants.csv"):
    print("\n➕ Add a new restaurant:")
    name = input("Restaurant Name: ")
    cuisine = input("Cuisine Type: ")
    rating = float(input("Rating (0-5): "))
    price = int(input("Price Level (1=cheap to 4=expensive): "))

    new_entry = pd.DataFrame([{
        "Name": name,
        "Cuisine": cuisine,
        "Rating": rating,
        "Price": price
    }])
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(file, index=False)
    print(f" '{name}' added to {file}.\n")

def show_restaurants(file="restaurants.csv"):
    if os.path.exists(file):
        df = pd.read_csv(file)
        if df.empty:
            print(" No restaurants saved yet.")
        else:
            print("\n Current Restaurants:")
            print(df)
    else:
        print(" No restaurants found.")

def remove_restaurant(file="restaurants.csv"):
    if not os.path.exists(file):
        print(" No restaurants to remove.")
        return
    df = pd.read_csv(file)
    if df.empty:
        print(" The list is empty.")
        return

    print("\n Select a restaurant to remove:")
    for i, row in df.iterrows():
        print(f"{i + 1}. {row['Name']} ({row['Cuisine']}, Rating: {row['Rating']})")

    try:
        idx = int(input("Enter the number to remove: ")) - 1
        if idx < 0 or idx >= len(df):
            print("Invalid number.")
            return
        removed = df.iloc[idx]
        df = df.drop(index=idx).reset_index(drop=True)
        df.to_csv(file, index=False)
        print(f" Removed '{removed['Name']}' from the list.")
    except ValueError:
        print("Please enter a valid number.")

def clear_restaurants(file="restaurants.csv"):
    confirm = input(" Are you sure you want to clear all restaurants? (y/n): ").strip().lower()
    if confirm == 'y':
        open(file, 'w').close()
        print("🧹 Restaurant list cleared.")
    else:
        print("Action canceled.")

def rate_restaurant(restaurants_file="restaurants.csv", history_file="user_data.csv"):
    print("\n Rate a restaurant you visited:")

    if os.path.exists(restaurants_file):
        restaurants = pd.read_csv(restaurants_file)
    else:
        restaurants = pd.DataFrame(columns=["Name", "Cuisine", "Rating", "Price"])

    if not restaurants.empty:
        for i, row in restaurants.iterrows():
            print(f"{i + 1}. {row['Name']} ({row['Cuisine']}, Rating: {row['Rating']})")

    choice = input("\nEnter number to select or type new restaurant name: ")

    selected = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(restaurants):
            selected = restaurants.iloc[idx].to_dict()
        else:
            print("Invalid selection.")
            return
    else:
        name = choice.strip()
        match = restaurants[restaurants["Name"].str.lower() == name.lower()]
        if not match.empty:
            selected = match.iloc[0].to_dict()
        else:
            cuisine = input(f"What is the cuisine of '{name}'? ")
            try:
                rating = float(input("Google/estimated rating (0-5): "))
                price = int(input("Price level (1 = cheap to 4 = expensive): "))
            except ValueError:
                print("❌ Invalid input for rating or price.")
                return
            selected = {
                "Name": name,
                "Cuisine": cuisine,
                "Rating": rating,
                "Price": price
            }
            restaurants = pd.concat([restaurants, pd.DataFrame([selected])], ignore_index=True)
            restaurants.to_csv(restaurants_file, index=False)
            print(f" '{name}' added to restaurant dataset.")

    try:
        dish = input("What did you eat? ")
        personal_rating = int(input("Rate the dish (1-10): "))
        liked = int(input("Did you like it? (1 = Yes, 0 = No): "))
        if liked not in [0, 1]:
            print("Invalid input. Use 1 or 0.")
            return
        date = input("Date visited (YYYY-MM-DD) [Leave blank for today]: ")
        if not date.strip():
            date = datetime.today().strftime('%Y-%m-%d')
    except ValueError:
        print(" Invalid input.")
        return

    entry = {
        "Name": selected["Name"],
        "Cuisine": selected["Cuisine"],
        "Rating": selected["Rating"],
        "Price": selected["Price"],
        "Liked": liked,
        "DateVisited": date,
        "Dish": dish,
        "PersonalRating": personal_rating
    }

    if os.path.exists("user_data.csv"):
        history = pd.read_csv("user_data.csv")
        history = pd.concat([history, pd.DataFrame([entry])], ignore_index=True)
    else:
        history = pd.DataFrame([entry])

    history.to_csv("user_data.csv", index=False)
    print(f" Dish rating for '{entry['Name']}' saved to history.")

def run_recommender():
    if not os.path.exists("user_data.csv") or not os.path.exists("restaurants.csv"):
        print(" Required files not found. Make sure 'user_data.csv' and 'restaurants.csv' exist.")
        return

    user_data = pd.read_csv("user_data.csv")
    new_places = pd.read_csv("restaurants.csv")

    recommender = FoodRecommender()
    recommender.fit(user_data)
    recommendations = recommender.predict(new_places)
    recommendations.to_csv("recommendations.csv", index=False)

    # Get top recommended restaurant
    top_restaurant = recommendations.iloc[0]
    cuisine = top_restaurant["Cuisine"]

    # Find matching high-rated dishes (score 8+)
    matching_dishes = user_data[
        (user_data["Cuisine"].str.lower() == cuisine.lower()) &
        (user_data["PersonalRating"] >= 8)
    ].sort_values("PersonalRating", ascending=False).head(3)

    print("\n🍴 Recommendation Summary:")
    print("🔹 Where to eat:")
    print(f"   → {top_restaurant['Name']} ({cuisine}) - Confidence: {top_restaurant['Confidence']:.2f}")

    print("🔹 What to eat:")
    if not matching_dishes.empty:
        for i, row in matching_dishes.iterrows():
            print(f"   • '{row['Dish']}' (Rated {row['PersonalRating']}/10)")
    else:
        print(f"   → No high-rated dishes yet for {cuisine}. Try something new!")

    print("\n Full recommendations saved to: recommendations.csv")

def show_menu():
    while True:
        print("\n===  Food Recommender Menu ===")
        print("1. Add a new restaurant")
        print("2. View saved restaurants")
        print("3. Get food recommendations")
        print("4. Rate a restaurant you visited")
        print("5. Remove a restaurant")
        print("6. Clear all restaurants")
        print("7. Exit")

        choice = input("Select an option (1-7): ")

        if choice == '1':
            add_restaurant_interactively()
        elif choice == '2':
            show_restaurants()
        elif choice == '3':
            run_recommender()
        elif choice == '4':
            rate_restaurant()
        elif choice == '5':
            remove_restaurant()
        elif choice == '6':
            clear_restaurants()
        elif choice == '7':
            print(" Bye!")
            break
        else:
            print("Invalid option. Try again.")

# Entry point
if __name__ == "__main__":
    show_menu()
