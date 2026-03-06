import pandas as pd
from src.predictor import simulate_match


def simulate_group_stage(groups, team_scores, team_stats):
    """
    Simulates the group stage of the tournament.
    Each team plays against every other team in their group.
    Win = 3 points, Loss = 0 points.
    """
    group_results = {}

    for group_name, teams in groups.items():
        standings = {team: {
            "points": 0,
            "sets_won": 0,
            "sets_lost": 0,
            "matches_won": 0,
            "matches_lost": 0
        } for team in teams}

        matches = []

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]

                result = simulate_match(team_a, team_b, team_scores, team_stats)
                matches.append(result)

                standings[team_a]["sets_won"] += result["sets_a"]
                standings[team_a]["sets_lost"] += result["sets_b"]
                standings[team_b]["sets_won"] += result["sets_b"]
                standings[team_b]["sets_lost"] += result["sets_a"]

                if result["sets_a"] > result["sets_b"]:
                    standings[team_a]["points"] += 3
                    standings[team_a]["matches_won"] += 1
                    standings[team_b]["matches_lost"] += 1
                else:
                    standings[team_b]["points"] += 3
                    standings[team_b]["matches_won"] += 1
                    standings[team_a]["matches_lost"] += 1

        standings_df = pd.DataFrame(standings).T.reset_index()
        standings_df.columns = ["Team", "Points", "Sets Won",
                                 "Sets Lost", "Matches Won", "Matches Lost"]
        standings_df["Sets Ratio"] = (
            standings_df["Sets Won"] / (standings_df["Sets Lost"] + 1)
        )
        standings_df = standings_df.sort_values(
            ["Points", "Sets Ratio"], ascending=False
        ).reset_index(drop=True)
        standings_df.index += 1

        group_results[group_name] = {
            "standings": standings_df,
            "matches": matches
        }

    return group_results


def simulate_knockout_stage(group_results, team_scores, team_stats):
    """
    Simulates knockout stage: quarterfinals, semifinals,
    third place match, and final.
    Top 2 teams from each group advance.
    """
    qualified = {}
    for group_name, results in group_results.items():
        standings = results["standings"]
        qualified[group_name] = {
            "first": standings.iloc[0]["Team"],
            "second": standings.iloc[1]["Team"]
        }

    quarterfinals = [
        (qualified["Group A"]["first"], qualified["Group B"]["second"]),
        (qualified["Group B"]["first"], qualified["Group A"]["second"]),
        (qualified["Group C"]["first"], qualified["Group D"]["second"]),
        (qualified["Group D"]["first"], qualified["Group C"]["second"]),
    ]

    qf_winners, qf_losers, qf_results = [], [], []
    for team_a, team_b in quarterfinals:
        result = simulate_match(team_a, team_b, team_scores, team_stats)
        winner = team_a if result["sets_a"] > result["sets_b"] else team_b
        loser = team_b if winner == team_a else team_a
        qf_winners.append(winner)
        qf_losers.append(loser)
        qf_results.append(result)

    semifinals = [
        (qf_winners[0], qf_winners[1]),
        (qf_winners[2], qf_winners[3])
    ]

    sf_winners, sf_losers, sf_results = [], [], []
    for team_a, team_b in semifinals:
        result = simulate_match(team_a, team_b, team_scores, team_stats)
        winner = team_a if result["sets_a"] > result["sets_b"] else team_b
        loser = team_b if winner == team_a else team_a
        sf_winners.append(winner)
        sf_losers.append(loser)
        sf_results.append(result)

    third_result = simulate_match(sf_losers[0], sf_losers[1], team_scores, team_stats)
    third_place = sf_losers[0] if third_result["sets_a"] > third_result["sets_b"] else sf_losers[1]
    fourth_place = sf_losers[1] if third_place == sf_losers[0] else sf_losers[0]

    final_result = simulate_match(sf_winners[0], sf_winners[1], team_scores, team_stats)
    champion = sf_winners[0] if final_result["sets_a"] > final_result["sets_b"] else sf_winners[1]
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third": third_place,
        "fourth": fourth_place,
        "qualified": qualified,
        "quarterfinals": {"matches": qf_results, "winners": qf_winners},
        "semifinals": {"matches": sf_results, "winners": sf_winners},
        "third_place_match": third_result,
        "final": final_result
    }