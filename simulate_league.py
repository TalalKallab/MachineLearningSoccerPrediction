import pandas as pd
from simulate_match import simulate_match

teams = [
    "Barcelona", "Real Madrid", "Ath Madrid", "Sevilla", "Valencia",
    "Villareal", "Betis", "Sociedad", "Ath Bilbao", "Espanol",
    "Getafe", "Osasuna", "Celta", "Vallecano", "Mallorca",
    "Cadiz", "Almeria", "Granada", "Las Palmas", "Girona"
]

league_table = {team: {"Points": 0, "Wins": 0, "Draws": 0, "Losses": 0, "GF": 0, "GA": 0, "GD": 0} for team in teams}

def simulate_league():
    print("\n Starting League Simulation...\n")
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                match_result = simulate_match(home_team, away_team)
                if match_result is None:
                    continue

                home_points = match_result["home_points"]
                away_points = match_result["away_points"]
                predicted_home_goals = match_result["predicted_home_goals"]
                predicted_away_goals = match_result["predicted_away_goals"]

                league_table[home_team]["Points"] += home_points
                league_table[away_team]["Points"] += away_points

                league_table[home_team]["GF"] += predicted_home_goals
                league_table[away_team]["GF"] += predicted_away_goals
                league_table[home_team]["GA"] += predicted_away_goals
                league_table[away_team]["GA"] += predicted_home_goals

                league_table[home_team]["GD"] = league_table[home_team]["GF"] - league_table[home_team]["GA"]
                league_table[away_team]["GD"] = league_table[away_team]["GF"] - league_table[away_team]["GA"]

                # Update Wins, Draws, and Losses
                if home_points == 3:
                    league_table[home_team]["Wins"] += 1
                    league_table[away_team]["Losses"] += 1
                elif away_points == 3:
                    league_table[away_team]["Wins"] += 1
                    league_table[home_team]["Losses"] += 1
                else:  # Draw case
                    league_table[home_team]["Draws"] += 1
                    league_table[away_team]["Draws"] += 1

    league_df = pd.DataFrame.from_dict(league_table, orient="index")
    league_df = league_df.sort_values(by=["Points", "GD", "GF"], ascending=[False, False, False])
    print("\n🏆 Final League Standings 🏆")
    print(league_df)
    return league_df

if __name__ == "__main__":
    final_table = simulate_league()
    print("\n League simulation completed successfully!")
