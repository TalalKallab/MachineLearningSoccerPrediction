import pandas as pd
import joblib

# Load the trained classification model at the start of the script
classification_model = joblib.load("models/match_result_model.pkl")


home_goals_model = joblib.load("home_goals_model.pkl")
away_goals_model = joblib.load("away_goals_model.pkl")

print("Models were loaded successfully")

def predict_result(home_team, away_team):
    # Step 1: Load the team mapping file
    file_path = "/Users/FILE_PATH/Team_Mapping.csv"
    data = pd.read_csv(file_path)

    # Load the main match data (LaLiga_V2.csv)
    file_path2 = "/Users/FILE_PATH/LaLiga_V2.csv"
    data2 = pd.read_csv(file_path2)

    # Load the season stats data (LaLiga_Season_Stats.csv)
    file_path3 = "/Users/FILE_PATH/LaLiga_Season_Stats.csv"
    data3 = pd.read_csv(file_path3)

    # Clean up column names to remove any extra whitespace
    data.columns = data.columns.str.strip()
    data2.columns = data2.columns.str.strip()
    data3.columns = data3.columns.str.strip()

    #Converting team names to numerical IDs using the mapping
    home_mapping = dict(zip(data['HomeTeam.1'], data['HomeTeam']))
    away_mapping = dict(zip(data['AwayTeam.1'], data['AwayTeam']))
    team_mapping = {**away_mapping, **home_mapping}

    # Check if both teams exist in the mapping dictionary
    if home_team not in team_mapping or away_team not in team_mapping:
        print(f"Error: One or both teams not found in the mapping file.")
        return None

    home_id = team_mapping[home_team]
    away_id = team_mapping[away_team]

    # Print the mapped IDs (for debugging purposes)
    print(f"Home Team: {home_team} -> ID: {home_id}")
    print(f"Away Team: {away_team} -> ID: {away_id}")

    #Prepare Input Features
    # Aggregate historical stats instead of using a single season
    champion = data3['Champion'].mode()[0]  # Most common champion
    champion_points = data3['ChampionPoints'].mean()  # Average points of champions
    top_scorer = data3['TopScorer'].mode()[0]  # Most common top scorer
    top_scorer_team = data3['TopScorer Team'].mode()[0]  # Team of most common top scorer
    most_assists = data3['MostAssists'].mode()[0]  # Most common assist leader
    most_assist_team = data3['MostAsistTeam'].mode()[0]  # Team of most common assist leader
    most_clean_sheets = data3['MostCleanSheets'].mode()[0]  # Most common clean sheet leader
    most_clean_sheets_team = data3['MostCleanSheetsTeam'].mode()[0]  # Team of most common clean sheet leader

    # Combine all features into a single feature vector (10 features as expected by the model)
    features = [
        home_id,              # Home team ID
        away_id,              # Away team ID
        champion,             # Most common champion team ID
        champion_points,      # Average points accumulated by champions
        top_scorer,           # Most common top scorer
        top_scorer_team,      # Team of the most common top scorer
        most_assists,         # Most common assist leader
        most_assist_team,     # Team of the most common assist leader
        most_clean_sheets,    # Most common clean sheet leader
        most_clean_sheets_team # Team of the most common clean sheet leader
    ]

    # Print the feature vector for debugging
    print("Feature Vector:", features)

    # Converting the feature vector into a DataFrame with correct column names
    features_df = pd.DataFrame([features], columns=[
        'HomeTeam', 'AwayTeam', 'Champion', 'ChampionPoints', 'TopScorer',
        'TopScorer Team', 'MostAssists', 'MostAsistTeam',
        'MostCleanSheets', 'MostCleanSheetsTeam'
    ])

    # Make the prediction
    predicted_label = classification_model.predict(features_df)[0]
    predicted_proba = classification_model.predict_proba(features_df)[0]

    #Interpret Prediction
    if predicted_label == 1:
        result = "Home Win"
    elif predicted_label == 0:
        result = "Draw"
    else:
        result = "Away Win"

    # Print the predicted label and probabilities
    print(f"Predicted Result: {result}")
    print(f"Prediction Probabilities - Home Win: {predicted_proba[0]:.2f}, Draw: {predicted_proba[1]:.2f}, Away Win: {predicted_proba[2]:.2f}")

    #Return or Print the Predicted Result
    final_result = f"Predicted Outcome: {result} (Confidence: Home Win: {predicted_proba[0]:.2f}, Draw: {predicted_proba[1]:.2f}, Away Win: {predicted_proba[2]:.2f})"
    print(final_result)

    return result, predicted_proba


predict_result("Barcelona", "Real Madrid")


def predict_goals(home_team, away_team):
    # Load the team mapping file
    file_path = "/Users/talalkallab/IdeaProjects/CEG4112_Project/Team_Mapping.csv"
    data = pd.read_csv(file_path)

    # Clean up the column names
    data.columns = data.columns.str.strip()

    # Converting team names to numerical IDs using the mapping
    home_mapping = dict(zip(data['HomeTeam.1'], data['HomeTeam']))
    away_mapping = dict(zip(data['AwayTeam.1'], data['AwayTeam']))
    team_mapping = {**away_mapping, **home_mapping}

    # Checking if both teams exist in the mapping CSV file
    if home_team not in team_mapping or away_team not in team_mapping:
        print(f"Error: One or both teams not found in the mapping file.")
        return None

    home_id = team_mapping[home_team]
    away_id = team_mapping[away_team]

    # Print the mapped IDs
    print(f"Home Team: {home_team} -> ID: {home_id}")
    print(f"Away Team: {away_team} -> ID: {away_id}")

    # Prepare the input features for goal prediction (use only relevant features)
    features = [home_id, away_id]

    # Convert the feature vector into a DataFrame with the correct column names
    features_df = pd.DataFrame([features], columns=['HomeTeam', 'AwayTeam'])

    # Print the DataFrame for verification
    print("Goal Prediction Feature DataFrame:")
    print(features_df)

    # Predict home and away goals
    predicted_home_goals = home_goals_model.predict(features_df)[0]
    predicted_away_goals = away_goals_model.predict(features_df)[0]

    # Rounding the predicted goals to make them more realistic
    predicted_home_goals = round(predicted_home_goals)
    predicted_away_goals = round(predicted_away_goals)

    # Print the predicted goals
    print(f"Predicted Goals - {home_team}: {predicted_home_goals}, {away_team}: {predicted_away_goals}")

    # Formatted result string
    result = f"Predicted Match: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}"
    print(result)

    return predicted_home_goals, predicted_away_goals, result

# Test the function
predict_goals("Celta", "Real Madrid")

def format_prediction(home_team, away_team, result, predicted_home_goals, predicted_away_goals, predicted_proba):
    #Create a header for the match prediction
    match_header = f"Predicted Match: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}"

    #Add the result of the match
    match_result = f"Predicted Result: {result}"

    #Add the confidence levels for each outcome
    confidence = (
        f"Confidence Levels:\n"
        f"  - Home Win: {predicted_proba[0] * 100:.1f}%\n"
        f"  - Draw: {predicted_proba[1] * 100:.1f}%\n"
        f"  - Away Win: {predicted_proba[2] * 100:.1f}%"
    )

    #Combine all parts into one formatted message
    formatted_output = f"{match_header}\n{match_result}\n{confidence}"

    return formatted_output
print("--")
# Get predictions
predicted_home_goals, predicted_away_goals, goal_prediction_result = predict_goals("Barcelona", "Real Madrid")
predicted_result, predicted_proba = predict_result("Barcelona", "Real Madrid")

# Combine and format the output
formatted_output = format_prediction(
    home_team="Barcelona",
    away_team="Real Madrid",
    result=predicted_result,
    predicted_home_goals=predicted_home_goals,
    predicted_away_goals=predicted_away_goals,
    predicted_proba=predicted_proba
)

# Print the nicely formatted result
print(formatted_output)
