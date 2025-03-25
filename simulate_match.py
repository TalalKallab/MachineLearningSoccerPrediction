from predict import predict_result, predict_goals, format_prediction

def simulate_match(home_team, away_team):

    try:

        predicted_result, predicted_proba = predict_result(home_team, away_team)
        if predicted_result is None or predicted_proba is None:
            print(f" Skipping match due to invalid team names or prediction errors: {home_team} vs {away_team}")
            return None

        predicted_home_goals, predicted_away_goals, _ = predict_goals(home_team, away_team)


        if predicted_result == "Home Win":
            home_points, away_points = 3, 0
        elif predicted_result == "Away Win":
            home_points, away_points = 0, 3
        else:  # Draw
            home_points, away_points = 1, 1


        print(f"️ Match: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team} | Result: {predicted_result}")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "predicted_result": predicted_result,
            "predicted_home_goals": predicted_home_goals,
            "predicted_away_goals": predicted_away_goals,
            "home_points": home_points,
            "away_points": away_points
        }

    except Exception as e:
        print(f"Error while simulating match {home_team} vs {away_team}: {str(e)}")
        return None
