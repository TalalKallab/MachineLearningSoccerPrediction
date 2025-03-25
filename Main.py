import os
import pandas as pd
from train_model import train_models
from simulate_league import simulate_league

def main():
    print("\n Starting the Soccer League Simulation Project...\n")


    print("🔧 Step 1: Training Models...")
    try:
        train_models()
        print(" Models trained successfully!\n")
    except Exception as e:
        print(f" Error during model training: {str(e)}")
        return


    print(" Step 2: Simulating the League...")
    try:
        final_table = simulate_league()
        print("\n Final League Standings:")
        print(final_table)
    except Exception as e:
        print(f" Error during league simulation: {str(e)}")
        return


    try:
        output_path = "final_league_standings.csv"
        final_table.to_csv(output_path)
        print(f"\n📂 Final standings saved as '{output_path}'")
    except Exception as e:
        print(f" Error saving final standings: {str(e)}")

    print("\n League simulation completed successfully!")

if __name__ == "__main__":
    main()
