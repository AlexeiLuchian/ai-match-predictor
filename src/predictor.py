import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_team_stats(df):
    """
    Aggregates individual player statistics per team
    and computes efficiency metrics.
    """
    team_stats = df.drop(columns=["Name"]).groupby("Team").agg({
        "Attack Points": "sum",
        "Attack Errors": "sum",
        "Attack Attempts": "sum",
        "Block Points": "sum",
        "Block Errors": "sum",
        "Rebounds": "sum",
        "Serve Points": "sum",
        "Serve Errors": "sum",
        "Serve Attempts": "sum",
        "Successful Sets": "sum",
        "Set Errors": "sum",
        "Set Attempts": "sum",
        "Spike Digs": "sum",
        "Dig Errors": "sum",
        "Successful Receives": "sum",
        "Receive Errors": "sum",
        "Receive Attempts": "sum",
    }).reset_index()

    team_stats["Attack Efficiency"] = team_stats["Attack Points"] / (team_stats["Attack Attempts"] + 1)
    team_stats["Serve Efficiency"] = team_stats["Serve Points"] / (team_stats["Serve Attempts"] + 1)
    team_stats["Block Efficiency"] = team_stats["Block Points"] / (team_stats["Block Errors"] + 1)

    return team_stats


def compute_strength_scores(team_stats):
    """
    Computes a strength score between 0 and 100 for each team
    based on weighted positive and negative statistics.
    """
    positive_stats = [
        "Attack Points", "Block Points", "Serve Points",
        "Rebounds", "Spike Digs", "Successful Receives",
        "Successful Sets", "Attack Efficiency", "Serve Efficiency",
        "Block Efficiency"
    ]
    negative_stats = [
        "Attack Errors", "Block Errors", "Serve Errors",
        "Set Errors", "Dig Errors", "Receive Errors"
    ]

    scaler = MinMaxScaler()
    team_scores = team_stats.copy()
    all_cols = positive_stats + negative_stats
    team_scores[all_cols] = scaler.fit_transform(team_stats[all_cols])

    team_scores["Strength Score"] = (
        team_scores[positive_stats].mean(axis=1) * 100 -
        team_scores[negative_stats].mean(axis=1) * 30
    )

    min_score = team_scores["Strength Score"].min()
    max_score = team_scores["Strength Score"].max()
    team_scores["Strength Score"] = (
        (team_scores["Strength Score"] - min_score) /
        (max_score - min_score) * 100
    )

    return team_scores


def simulate_match(team_a, team_b, team_scores, team_stats):
    """
    Simulates a volleyball match between two teams.
    Returns sets won by each team, set scores, and win probability.
    """
    score_a = team_scores.loc[team_scores["Team"] == team_a, "Strength Score"].values[0]
    score_b = team_scores.loc[team_scores["Team"] == team_b, "Strength Score"].values[0]

    diff = score_a - score_b
    prob_a_wins = 1 / (1 + np.exp(-diff / 20))

    sets_a, sets_b = 0, 0
    set_scores = []

    while sets_a < 3 and sets_b < 3:
        is_deciding_set = (sets_a + sets_b == 4)
        max_points = 15 if is_deciding_set else 25

        if np.random.random() < prob_a_wins:
            points_a = max_points
            points_b = np.random.randint(max_points - 10, max_points - 1)
            sets_a += 1
        else:
            points_b = max_points
            points_a = np.random.randint(max_points - 10, max_points - 1)
            sets_b += 1

        set_scores.append((points_a, points_b))

    return {
        "team_a": team_a,
        "team_b": team_b,
        "sets_a": sets_a,
        "sets_b": sets_b,
        "set_scores": set_scores,
        "prob_a_wins": round(prob_a_wins * 100, 1)
    }