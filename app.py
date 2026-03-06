# app.py

import streamlit as st
import pandas as pd
from src.predictor import compute_team_stats, compute_strength_scores
from src.tournament import simulate_group_stage, simulate_knockout_stage
from src.shap_explainer import train_explainer, explain_match
from src.nlp import generate_match_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VNL Tournament Simulator",
    page_icon="🏐",
    layout="wide"
)

# ── Team flags ────────────────────────────────────────────────────────────────
FLAG_CODES = {
    "ARG": "ar", "BRA": "br", "BUL": "bg", "CAN": "ca",
    "CHN": "cn", "CUB": "cu", "FRA": "fr", "GER": "de",
    "IRI": "ir", "ITA": "it", "JPN": "jp", "NED": "nl",
    "POL": "pl", "SLO": "si", "SRB": "rs", "USA": "us"
}

def flag(team):
    code = FLAG_CODES.get(team, "")
    if code:
        img = f'<img src="https://flagcdn.com/20x15/{code}.png" style="vertical-align:middle; margin-right:4px;">'
        return f'{img}{team}'
    return team

def flag_text(team):
    """Plain text version for selectbox and non-HTML contexts."""
    code = FLAG_CODES.get(team, "")
    flag_emojis = {
        "ar": "🇦🇷", "br": "🇧🇷", "bg": "🇧🇬", "ca": "🇨🇦",
        "cn": "🇨🇳", "cu": "🇨🇺", "fr": "🇫🇷", "de": "🇩🇪",
        "ir": "🇮🇷", "it": "🇮🇹", "jp": "🇯🇵", "nl": "🇳🇱",
        "pl": "🇵🇱", "si": "🇸🇮", "rs": "🇷🇸", "us": "🇺🇸"
    }
    return f"{flag_emojis.get(code, '')} {team}"

# ── Load and prepare data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")
    team_stats = compute_team_stats(df)
    team_scores = compute_strength_scores(team_stats)
    return team_stats, team_scores

@st.cache_resource
def load_explainer(team_stats, team_scores):
    explainer, model, feature_cols = train_explainer(team_stats, team_scores)
    return explainer, model, feature_cols

team_stats, team_scores = load_data()
explainer, xgb_model, feature_cols = load_explainer(team_stats, team_scores)

# ── Session state init ────────────────────────────────────────────────────────
if "group_results" not in st.session_state:
    st.session_state.group_results = None
if "knockout_results" not in st.session_state:
    st.session_state.knockout_results = None
if "all_matches" not in st.session_state:
    st.session_state.all_matches = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏐 VNL Tournament Simulator")
st.sidebar.markdown("Simulate a full volleyball tournament with AI-powered match analysis.")
st.sidebar.subheader("Tournament Groups")

all_teams = sorted(team_stats["Team"].tolist())

default_groups = {
    "Group A": ["BRA", "USA", "POL", "IRI"],
    "Group B": ["ITA", "FRA", "ARG", "JPN"],
    "Group C": ["GER", "SRB", "NED", "SLO"],
    "Group D": ["CUB", "CAN", "BUL", "CHN"]
}

groups = {}
for group_name, default_teams in default_groups.items():
    selected = st.sidebar.multiselect(
        group_name,
        options=all_teams,
        default=default_teams,
        max_selections=4
    )
    groups[group_name] = selected

simulate_btn = st.sidebar.button("Simulate Tournament", use_container_width=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏐 VNL Tournament Simulator")

# Team strength rankings
st.subheader("Team Strength Rankings")
rankings = team_scores[["Team", "Strength Score"]].copy()
rankings["Strength Score"] = rankings["Strength Score"].round(2)
rankings = rankings.sort_values("Strength Score", ascending=False).reset_index(drop=True)
rankings.index += 1
st.dataframe(rankings, use_container_width=True)

# ── Simulate ──────────────────────────────────────────────────────────────────
if simulate_btn:
    all_selected = [t for g in groups.values() for t in g]
    if len(all_selected) != len(set(all_selected)):
        st.error("Each team can only appear in one group.")
    elif any(len(g) != 4 for g in groups.values()):
        st.error("Each group must have exactly 4 teams.")
    else:
        with st.spinner("Simulating group stage..."):
            st.session_state.group_results = simulate_group_stage(
                groups, team_scores, team_stats
            )
        with st.spinner("Simulating knockout stage..."):
            st.session_state.knockout_results = simulate_knockout_stage(
                st.session_state.group_results, team_scores, team_stats
            )

        all_matches = []
        for group_name, results in st.session_state.group_results.items():
            for match in results["matches"]:
                all_matches.append((
                    f"{group_name}: {flag_text(match['team_a'])} vs {flag_text(match['team_b'])}",
                    match
                ))
        for match in st.session_state.knockout_results["quarterfinals"]["matches"]:
            all_matches.append((
                f"QF: {flag_text(match['team_a'])} vs {flag_text(match['team_b'])}",
                match
            ))
        for match in st.session_state.knockout_results["semifinals"]["matches"]:
            all_matches.append((
                f"SF: {flag_text(match['team_a'])} vs {flag_text(match['team_b'])}",
                match
            ))
        m = st.session_state.knockout_results["third_place_match"]
        all_matches.append((f"3rd Place: {flag_text(m['team_a'])} vs {flag_text(m['team_b'])}", m))
        m = st.session_state.knockout_results["final"]
        all_matches.append((f"Final: {flag_text(m['team_a'])} vs {flag_text(m['team_b'])}", m))

        st.session_state.all_matches = all_matches

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.group_results:
    group_results = st.session_state.group_results
    knockout_results = st.session_state.knockout_results

    # Group stage
    st.header("Group Stage")
    for group_name, results in group_results.items():
        st.subheader(group_name)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Standings**")
            st.dataframe(
                results["standings"][["Team", "Points", "Matches Won", "Matches Lost", "Sets Won", "Sets Lost"]],
                use_container_width=True
            )

        with col2:
            st.markdown("**Match Results**")
            for match in results["matches"]:
                winner = match["team_a"] if match["sets_a"] > match["sets_b"] else match["team_b"]
                set_scores_str = " | ".join([f"{a}-{b}" for a, b in match["set_scores"]])
                st.markdown(
                    f"{flag(match['team_a'])} vs {flag(match['team_b'])} &nbsp;→&nbsp; "
                    f"**{match['sets_a']}:{match['sets_b']}** sets &nbsp;({set_scores_str})&nbsp; "
                    f"— winner: **{flag(winner)}**",
                    unsafe_allow_html=True
                )

    # Knockout stage
    st.header("Knockout Stage")

    st.subheader("Qualified Teams")
    qual_cols = st.columns(4)
    for i, (group_name, teams) in enumerate(knockout_results["qualified"].items()):
        with qual_cols[i]:
            st.markdown(f"**{group_name}**")
            st.markdown(f"🥇 {flag(teams['first'])}", unsafe_allow_html=True)
            st.markdown(f"🥈 {flag(teams['second'])}", unsafe_allow_html=True)

    st.subheader("Quarterfinals")
    for i, match in enumerate(knockout_results["quarterfinals"]["matches"]):
        winner = knockout_results["quarterfinals"]["winners"][i]
        set_scores_str = " | ".join([f"{a}-{b}" for a, b in match["set_scores"]])
        st.markdown(
            f"{flag(match['team_a'])} vs {flag(match['team_b'])} &nbsp;→&nbsp; "
            f"**{match['sets_a']}:{match['sets_b']}** sets &nbsp;({set_scores_str})&nbsp; "
            f"— winner: **{flag(winner)}**",
            unsafe_allow_html=True
        )

    st.subheader("Semifinals")
    for i, match in enumerate(knockout_results["semifinals"]["matches"]):
        winner = knockout_results["semifinals"]["winners"][i]
        set_scores_str = " | ".join([f"{a}-{b}" for a, b in match["set_scores"]])
        st.markdown(
            f"{flag(match['team_a'])} vs {flag(match['team_b'])} &nbsp;→&nbsp; "
            f"**{match['sets_a']}:{match['sets_b']}** sets &nbsp;({set_scores_str})&nbsp; "
            f"— winner: **{flag(winner)}**",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Third Place Match")
        m = knockout_results["third_place_match"]
        winner = knockout_results["third"]
        set_scores_str = " | ".join([f"{a}-{b}" for a, b in m["set_scores"]])
        st.markdown(
            f"{flag(m['team_a'])} vs {flag(m['team_b'])} &nbsp;→&nbsp; "
            f"**{m['sets_a']}:{m['sets_b']}** sets &nbsp;({set_scores_str})&nbsp; "
            f"— 🥉 **{flag(winner)}**",
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Final")
        m = knockout_results["final"]
        winner = knockout_results["champion"]
        set_scores_str = " | ".join([f"{a}-{b}" for a, b in m["set_scores"]])
        st.markdown(
            f"{flag(m['team_a'])} vs {flag(m['team_b'])} &nbsp;→&nbsp; "
            f"**{m['sets_a']}:{m['sets_b']}** sets &nbsp;({set_scores_str})&nbsp; "
            f"— 🏆 **{flag(winner)}**",
            unsafe_allow_html=True
        )

    # Final standings
    st.header("Final Standings")
    st.markdown(f"🥇 **1st:** {flag(knockout_results['champion'])}", unsafe_allow_html=True)
    st.markdown(f"🥈 **2nd:** {flag(knockout_results['runner_up'])}", unsafe_allow_html=True)
    st.markdown(f"🥉 **3rd:** {flag(knockout_results['third'])}", unsafe_allow_html=True)
    st.markdown(f"**4th:** {flag(knockout_results['fourth'])}", unsafe_allow_html=True)

    # AI Match Analysis
    st.header("AI Match Analysis")
    st.markdown("Select any match to get an AI-generated analysis powered by SHAP explainability and a HuggingFace language model.")

    selected_match_label = st.selectbox(
        "Choose a match to analyze:",
        options=[label for label, _ in st.session_state.all_matches]
    )

    selected_match = next(
        match for label, match in st.session_state.all_matches
        if label == selected_match_label
    )

    if st.button("Generate Analysis"):
        with st.spinner("Calculating SHAP values and generating analysis..."):
            shap_vals = explain_match(
                selected_match["team_a"],
                selected_match["team_b"],
                team_stats,
                explainer,
                feature_cols
            )
            analysis = generate_match_analysis(
                selected_match["team_a"],
                selected_match["team_b"],
                selected_match,
                shap_vals,
                feature_cols
            )

        winner = selected_match["team_a"] if selected_match["sets_a"] > selected_match["sets_b"] else selected_match["team_b"]

        st.markdown(
            f"### {flag(selected_match['team_a'])} vs {flag(selected_match['team_b'])}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Result:** {selected_match['sets_a']}:{selected_match['sets_b']} — winner: **{flag(winner)}**",
            unsafe_allow_html=True
        )

        st.markdown("**Key statistical factors (SHAP):**")
        shap_importance = sorted(
            zip(feature_cols, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        for feat, val in shap_importance:
            direction = f"favored {flag_text(selected_match['team_a'])}" if val > 0 else f"favored {flag_text(selected_match['team_b'])}"
            st.markdown(f"- **{feat}**: {val:.4f} ({direction})")

        st.markdown("**AI Analysis:**")
        st.info(analysis)