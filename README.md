# MachineLearningSoccerPrediction

Project Overview
This project leverages machine learning to predict and simulate a full soccer league season (specifically La Liga) using historical data. The system predicts match outcomes (win, draw, loss) and goals scored by each team using Random Forest Classifier and Random Forest Regressor models. The project also generates simulated league standings, which are exported as CSV and PDF reports.

Features
Match Outcome Prediction: Predicts whether the home team wins, the away team wins, or the match is a draw.
Goal Prediction: Estimates the number of goals scored by both the home and away teams.
League Simulation: Simulates an entire soccer league season and produces final league standings.
Statistical Reports: Exports league standings and predictions to CSV and PDF formats.

Machine Learning Approach

Algorithms Used:
Random Forest Classifier: To predict match outcomes (win, draw, loss).
Random Forest Regressor: To predict the number of goals scored by each team.

Model Training:
Trained on historical match data from La Liga covering multiple seasons.
Data collected from reliable sources such as Transfermarkt, FBRef, and Kaggle.

Performance Metrics:
Accuracy: Evaluated using Accuracy Score for classification and Mean Absolute Error (MAE) for regression.

Data Sources
The project uses historical data from the following sources:

Transfermarkt La Liga Clean Sheets --> https://www.transfermarkt.com/laliga/weisseWeste/wettbewerb/ES1/saison_id/1995#google_vignette

FBRef La Liga Stats (1995-2023) --> https://fbref.com/en/comps/12/2016-2017/2016-2017-La-Liga-Stats

Kaggle La Liga Results (1995-2020) --> https://www.kaggle.com/datasets/kishan305/la-liga-results-19952020

FBRef La Liga Seasons History (1995-2024) --> https://fbref.com/en/comps/12/history/La-Liga-Seasons


Python: Core programming language.

Scikit-learn: For training classification and regression models.

Pandas: Data manipulation and cleaning.

Joblib: Model serialization and persistence.



