import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def train_explainer(team_stats, team_scores, n_simulations=500):
    """
    Trains an XGBoost model on synthetic match data and returns
    a fitted SHAP TreeExplainer along with the feature column names.
    """
    from src.predictor import simulate_match

    feature_cols = [
        "Attack Points", "Block Points", "Serve Points",
        "Rebounds", "Spike Digs", "Successful Receives",
        "Attack Efficiency", "Serve Efficiency", "Block Efficiency",
        "Attack Errors", "Block Errors", "Serve Errors"
    ]

    teams = team_stats["Team"].tolist()
    X, y = [], []

    for _ in range(n_simulations):
        team_a, team_b = np.random.choice(teams, 2, replace=False)

        stats_a = team_stats[team_stats["Team"] == team_a][feature_cols].values[0]
        stats_b = team_stats[team_stats["Team"] == team_b][feature_cols].values[0]

        diff = stats_a - stats_b
        X.append(diff)

        result = simulate_match(team_a, team_b, team_scores, team_stats)
        y.append(1 if result["sets_a"] > result["sets_b"] else 0)

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    return explainer, model, feature_cols


def explain_match(team_a, team_b, team_stats, explainer, feature_cols):
    """
    Returns SHAP values for a specific matchup between two teams.
    Positive values favor team_a, negative values favor team_b.
    """
    stats_a = team_stats[team_stats["Team"] == team_a][feature_cols].values[0]
    stats_b = team_stats[team_stats["Team"] == team_b][feature_cols].values[0]

    diff = (stats_a - stats_b).reshape(1, -1)
    shap_vals = explainer.shap_values(diff)[0]

    return shap_vals