import pandas as pd
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# --- Load Data ---
def load_data():
    file_path = "/Users/talalkallab/IdeaProjects/CEG4112_Project/LaLiga_V2.csv"
    data = pd.read_csv(file_path)
    season_stats_path = "/Users/talalkallab/IdeaProjects/CEG4112_Project/LaLiga_Season_Stats.csv"
    season_stats = pd.read_csv(season_stats_path)

    print("First 5 Rows of Main Data:")
    print(data.head())
    print("\nFirst 5 Rows of Season Stats Data:")
    print(season_stats.head())

    print("\nMissing Values in Main Data:")
    print(data.isnull().sum())
    print("\nMissing Values in Season Stats Data:")
    print(season_stats.isnull().sum())

    return data, season_stats

data, season_stats = load_data()
print("**" * 80)

# --- Extract Season from Match Dates ---
def extract_season(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Season'] = data['Date'].apply(
        lambda x: f"{x.year}-{x.year + 1}" if x.month >= 8 else f"{x.year - 1}-{x.year}"
    )
    return data

data = extract_season(data)

# --- Merge Datasets ---
def merge_datasets(data, season_stats):
    merged_data = data.merge(season_stats, on='Season', how='left')
    return merged_data

merged_data = merge_datasets(data, season_stats)

# --- Data Splitting ---
def split_data(merged_data):
    X = merged_data[['HomeTeam', 'AwayTeam', 'Champion', 'ChampionPoints', 'TopScorer', 'TopScorer Team',
                     'MostAssists', 'MostAsistTeam', 'MostCleanSheets', 'MostCleanSheetsTeam']]
    y = merged_data['Winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_home = merged_data['HomeGoals']
    y_away = merged_data['AwayGoals']



    X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42)
    X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X_train_home, X_test_home, y_train_home, y_test_home, X_train_away, X_test_away, y_train_away, y_test_away

X_train, X_test, y_train, y_test, X_train_home, X_test_home, y_train_home, y_test_home, X_train_away, X_test_away, y_train_away, y_test_away = split_data(merged_data)
print(X_train.head())

# --- Save Models ---
def save_models(classification_model, home_goal_model, away_goal_model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(classification_model, "models/match_result_model.pkl")
    joblib.dump(home_goal_model, "models/home_goals_model.pkl")
    joblib.dump(away_goal_model, "models/away_goals_model.pkl")
    print("Models saved successfully.")

# --- Model Training ---
def train_models():
    print("Training Classification Model (Match Result Prediction)...")
    model_classification = RandomForestClassifier(n_estimators=200, random_state=42)
    model_classification.fit(X_train, y_train)
    y_pred_classification = model_classification.predict(X_test)
    accuracy_classification = accuracy_score(y_test, y_pred_classification)
    print("Classification Accuracy: ", accuracy_classification)

    print("Training Regression Model (Home Goals Prediction)...")
    model_home_goals = RandomForestRegressor(n_estimators=200, random_state=42)
    model_home_goals.fit(X_train_home, y_train_home)
    y_pred_home = model_home_goals.predict(X_test_home)
    MAE_HOME = abs(y_pred_home - y_test_home).mean()
    print("Mean Absolute Error (HomeGOALS): ", MAE_HOME)

    print("Training Regression Model (Away Goals Prediction)...")
    model_away_goals = RandomForestRegressor(n_estimators=200, random_state=42)
    model_away_goals.fit(X_train_away, y_train_away)
    y_pred_away = model_away_goals.predict(X_test_away)
    MAE_AWAY = abs(y_pred_away - y_test_away).mean()
    print("Mean Absolute Error (AwayGOALS): ", MAE_AWAY)

    print("Home Goals Model Features: ", X_train_home.columns)
    print("Away Goals Model Features: ", X_train_away.columns)

    save_models(model_classification, model_home_goals, model_away_goals)

train_models()

# --- Evaluate Models ---
def evaluate_models():
    match_result_model = joblib.load("models/match_result_model.pkl")
    home_goal_model = joblib.load("models/home_goals_model.pkl")
    away_goal_model = joblib.load("models/away_goals_model.pkl")

    (X_train, X_test, y_train, y_test, X_train_home, X_test_home, y_train_home, y_test_home,
     X_train_away, X_test_away, y_train_away, y_test_away) = split_data(merged_data)

    print("\n--- Evaluating Models ---")
    y_pred_classification = match_result_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_classification)
    print("Classification ACCURACY (MATCH RESULT PREDICTION): ", accuracy)

    y_pred_home = home_goal_model.predict(X_test_home)
    MAE_HOME = abs(y_pred_home - y_test_home).mean()
    print("Mean Absolute Error (HomeGOALS): ", MAE_HOME)

    y_pred_away = away_goal_model.predict(X_test_away)
    MAE_AWAY = abs(y_pred_away - y_test_away).mean()
    print("Mean Absolute Error (AwayGOALS): ", MAE_AWAY)

    print("\n--- SAMPLE PREDICTION VS ACTUAL PREDICTION ---")
    for i in range(10):
        print(f"Match: {X_test.iloc[i]['HomeTeam']} vs {X_test.iloc[i]['AwayTeam']}")
        print(f"  Predicted Result: {y_pred_classification[i]}, Actual Result: {y_test.iloc[i]}")
        print(f"  Predicted Home Goals: {y_pred_home[i]:.1f}, Actual Home Goals: {y_test_home.iloc[i]}")
        print(f"  Predicted Away Goals: {y_pred_away[i]:.1f}, Actual Away Goals: {y_test_away.iloc[i]}")
        print("-" * 40)

    print("Model evaluation completed successfully.")

evaluate_models()
