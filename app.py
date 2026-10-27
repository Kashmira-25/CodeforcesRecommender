
from pathlib import Path
import ast
import re
import time

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Codeforces Problem Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
API = "https://codeforces.com/api"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CodeforcesProblemCoach/1.0"})


# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #f5f8ff;
    }
    [data-testid="stHeader"] {
        background: rgba(245,248,255,0.92);
    }
    [data-testid="stSidebar"] {
        background: #edf3ff;
        border-right: 1px solid #dce7fa;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }
    .hero {
        background: linear-gradient(120deg, #173c75, #2679ef);
        color: white;
        padding: 2rem 2.2rem;
        border-radius: 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 28px rgba(31,83,160,.14);
    }
    .hero h1 {
        color: white;
        font-size: 2.45rem;
        line-height: 1.15;
        margin: .3rem 0 .7rem 0;
    }
    .hero p {
        color: #e5efff;
        font-size: 1.05rem;
        margin-bottom: .2rem;
    }
    .eyebrow {
        color: #c8ddff;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .section-title {
        color: #193b70;
        font-size: 1.35rem;
        font-weight: 750;
        margin: 1rem 0 .6rem 0;
    }
    .mini-note {
        color: #64748b;
        font-size: .9rem;
    }
    .problem-card {
        background: white;
        border: 1px solid #dfe9fa;
        border-radius: 15px;
        padding: 1rem 1.1rem;
        margin: .65rem 0;
    }
    .problem-title {
        color: #173c75;
        font-size: 1.05rem;
        font-weight: 750;
    }
    .tag-pill {
        display: inline-block;
        background: #e9f1ff;
        color: #2459a8;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: .76rem;
        margin: 4px 4px 2px 0;
    }
    .mode-card {
        background: white;
        border: 1px solid #dfe9fa;
        padding: 1rem;
        border-radius: 16px;
        height: 100%;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe9fa;
        border-radius: 15px;
        padding: 13px 16px;
    }
    .stButton button[kind="primary"] {
        border-radius: 10px;
        font-weight: 700;
    }
    a {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA HELPERS
# --------------------------------------------------
def find_file(filename):
    matches = list(ROOT.rglob(filename))
    return matches[0] if matches else None


def clean_id(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def parse_tags(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    return [
        x.strip().strip("'\"[] ")
        for x in re.split(r"[,;]", text)
        if x.strip().strip("'\"[] ")
    ]


@st.cache_data(ttl=300, show_spinner=False)
def load_dataset():
    interaction_file = find_file("processed_data.csv")
    problems_file = find_file("problems.csv")

    if interaction_file is None:
        raise FileNotFoundError(
            "processed_data.csv was not found. Keep your dataset folder "
            "inside the project folder."
        )

    interactions = pd.read_csv(interaction_file, low_memory=False)
    interactions.columns = interactions.columns.str.strip().str.lower()

    required = {"user", "problem_id", "solved"}
    if not required.issubset(interactions.columns):
        raise ValueError(
            f"processed_data.csv needs columns: {sorted(required)}"
        )

    interactions["user"] = interactions["user"].astype(str)
    interactions["problem_id"] = interactions["problem_id"].astype(str)
    interactions["join_id"] = interactions["problem_id"].map(clean_id)
    interactions["solved"] = pd.to_numeric(
        interactions["solved"], errors="coerce"
    ).fillna(0)

    if "rating" in interactions:
        interactions["rating"] = pd.to_numeric(
            interactions["rating"], errors="coerce"
        )
    else:
        interactions["rating"] = pd.NA

    if "tags" not in interactions:
        interactions["tags"] = ""

    interactions["tag_list"] = interactions["tags"].apply(parse_tags)

    if problems_file:
        problems = pd.read_csv(problems_file, low_memory=False)
        problems.columns = problems.columns.str.strip().str.lower()
    else:
        problems = pd.DataFrame()

    if not problems.empty:
        if "problem_id" not in problems.columns:
            contest_col = next(
                (c for c in ["contestid", "contest_id"] if c in problems),
                None,
            )
            if contest_col and "index" in problems:
                problems["problem_id"] = (
                    problems[contest_col].astype(str)
                    + "-"
                    + problems["index"].astype(str)
                )

        if "problem_id" in problems:
            problems["problem_id"] = problems["problem_id"].astype(str)
            problems["join_id"] = problems["problem_id"].map(clean_id)

        if "rating" in problems:
            problems["rating"] = pd.to_numeric(
                problems["rating"], errors="coerce"
            )
        else:
            problems["rating"] = pd.NA

        if "tags" not in problems:
            problems["tags"] = ""

        problems["tag_list"] = problems["tags"].apply(parse_tags)

    return interactions, problems


def cf_request(method, params):
    try:
        response = SESSION.get(
            f"{API}/{method}", params=params, timeout=20
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Codeforces. Check your internet connection "
            "and try again."
        ) from exc

    if data.get("status") != "OK":
        raise RuntimeError(
            data.get("comment", "Codeforces API returned an error.")
        )

    return data["result"]


def fetch_live_profile(handle):
    profile = cf_request("user.info", {"handles": handle})[0]

    # Fetch recent submissions. Codeforces API limits this request to
    # the most recent 10,000 submissions.
    submissions = cf_request(
        "user.status", {"handle": handle, "from": 1, "count": 10000}
    )

    return profile, submissions


def get_live_problems():
    result = cf_request("problemset.problems", {})
    problems = result.get("problems", [])
    stats = result.get("problemStatistics", [])

    stat_lookup = {}
    for stat in stats:
        key = (stat.get("contestId"), stat.get("index"))
        stat_lookup[key] = stat.get("solvedCount", 0)

    records = []
    for problem in problems:
        contest = problem.get("contestId")
        index = problem.get("index", "")
        pid = f"{contest}-{index}" if contest is not None else index

        records.append({
            "problem_id": pid,
            "join_id": clean_id(pid),
            "name": problem.get("name", "Unnamed problem"),
            "rating": problem.get("rating"),
            "tag_list": problem.get("tags", []),
            "solved_count": stat_lookup.get((contest, index), 0),
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# RECOMMENDATION ENGINE
# --------------------------------------------------
def rank_problems(candidates, attempted, solved, weak_topics, target_rating):
    candidates = candidates.copy()

    candidates = candidates[
        ~candidates["join_id"].isin(solved)
        & ~candidates["join_id"].isin(attempted)
    ].copy()

    if candidates.empty:
        return candidates

    def score_problem(row):
        tags = row.get("tag_list", [])
        if not isinstance(tags, list):
            tags = parse_tags(tags)

        topic_score = (
            sum(weak_topics.get(tag, 0.2) for tag in tags) / len(tags)
            if tags else 0.2
        )

        rating = row.get("rating")
        if pd.notna(rating):
            rating_score = max(
                0.0, 1.0 - abs(float(rating) - target_rating) / 1000
            )
        else:
            rating_score = 0.35

        popularity = row.get("solved_count", 0)
        if pd.isna(popularity):
            popularity = 0
        popularity_score = min(float(popularity) / 50000, 1.0)

        return (
            0.55 * topic_score
            + 0.35 * rating_score
            + 0.10 * popularity_score
        )

    candidates["score"] = candidates.apply(score_problem, axis=1)
    return candidates.sort_values("score", ascending=False)


def dataset_topic_weakness(user_data):
    stats = {}
    for _, row in user_data.iterrows():
        for tag in row.get("tag_list", []):
            stats.setdefault(tag, {"attempts": 0, "solved": 0})
            stats[tag]["attempts"] += 1
            stats[tag]["solved"] += int(row["solved"] > 0)

    return {
        tag: 1 - v["solved"] / v["attempts"]
        for tag, v in stats.items()
        if v["attempts"] >= 2
    }


def live_topic_weakness(submissions):
    stats = {}
    for sub in submissions:
        problem = sub.get("problem", {})
        tags = problem.get("tags", [])
        verdict = sub.get("verdict")
        for tag in tags:
            stats.setdefault(tag, {"attempts": 0, "solved": 0})
            stats[tag]["attempts"] += 1
            stats[tag]["solved"] += int(verdict == "OK")

    return {
        tag: 1 - v["solved"] / v["attempts"]
        for tag, v in stats.items()
        if v["attempts"] >= 2
    }


def render_recommendations(results, limit=10):
    st.markdown('<div class="section-title">🏆 Recommended problems</div>',
                unsafe_allow_html=True)
    st.caption(
        "Higher ranking scores indicate a better match under this "
        "demo ranking method. Scores are not probabilities."
    )

    if results.empty:
        st.warning(
            "No eligible problems were found. Try another user or a "
            "different rating range."
        )
        return

    for rank, (_, row) in enumerate(results.head(limit).iterrows(), 1):
        pid = str(row.get("problem_id", ""))
        name = str(row.get("name", f"Problem {pid}"))
        rating = row.get("rating")
        rating_label = (
            str(int(rating)) if pd.notna(rating) else "Unrated"
        )
        tags = row.get("tag_list", [])
        if not isinstance(tags, list):
            tags = parse_tags(tags)

        tags_html = "".join(
            f'<span class="tag-pill">{tag}</span>'
            for tag in tags[:6]
        )
        contest_match = re.fullmatch(r"(\d+)-?([A-Z]\d*)", pid.upper())
        link = ""
        if contest_match:
            cid, idx = contest_match.groups()
            link = (
                f"https://codeforces.com/problemset/problem/{cid}/{idx}"
            )

        st.markdown(
            f"""
            <div class="problem-card">
              <div class="problem-title">{rank}. {name}</div>
              <div style="color:#64748b;font-size:.88rem;margin-top:4px">
                Problem ID: {pid} &nbsp;·&nbsp; Rating: {rating_label}
              </div>
              <div style="margin-top:7px">{tags_html}</div>
              <div style="margin-top:9px;color:#2459a8;font-size:.9rem">
                Ranking score: {row["score"]:.3f}
              </div>
              {"<a href='" + link + "' target='_blank'>Open on Codeforces ↗</a>" if link else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Problem Coach")
    st.caption("Smarter practice. Better progress.")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "Recommendations", "Topic Analysis", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Why use this?")
    st.write("🎯 Practice at a suitable difficulty")
    st.write("🧠 Find topics to improve")
    st.write("📊 Explore your practice history")
    st.write("🧪 Try a demo without an account")

    st.divider()
    st.caption("Powered by Codeforces data and a practice-ranking prototype.")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
try:
    with st.spinner("Loading the recommendation dataset..."):
        interactions, catalog = load_dataset()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.info(
        "Check that processed_data.csv exists somewhere inside "
        "your CodeforcesRecommender folder."
    )
    st.stop()


# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">YOUR PERSONAL PRACTICE COMPANION</div>
      <h1>Codeforces Problem Coach ⚡</h1>
      <p>Find your next challenge, strengthen weak topics, and practice smarter.</p>
      <p>Choose demo mode or use your own public Codeforces submission history.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Choose how you want to get started")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="mode-card">
          <h3>🧪 Explore Demo Mode</h3>
          <p>Don't have a Codeforces account? No problem.</p>
          <p>Use a sample user from the research dataset to explore
          recommendations and topic analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Demo →", type="primary", use_container_width=True):
            st.session_state["mode"] = "Demo"
            st.session_state["page"] = "Recommendations"
            st.rerun()

    with right:
        st.markdown("""
        <div class="mode-card">
          <h3>👤 Use Your Codeforces Handle</h3>
          <p>Already use Codeforces? Enter your public handle to
          analyze your recent submissions and find new problems.</p>
          <p>No password or API key is needed.</p>
        </div>
        """, unsafe_allow_html=True)
        handle_input = st.text_input(
            "Your Codeforces handle",
            placeholder="e.g. tourist",
            key="home_handle",
        )
        if st.button(
            "Get My Recommendations →",
            use_container_width=True,
            type="primary",
        ):
            if handle_input.strip():
                st.session_state["mode"] = "Live"
                st.session_state["handle"] = handle_input.strip()
                st.session_state["page"] = "Recommendations"
                st.rerun()
            else:
                st.warning("Enter your Codeforces handle first.")

    st.markdown("---")
    st.markdown("### 📊 What's inside")

    total_users = interactions["user"].nunique()
    total_interactions = len(interactions)
    total_problems = (
        catalog["join_id"].nunique()
        if not catalog.empty and "join_id" in catalog else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset interactions", f"{total_interactions:,}")
    c2.metric("Dataset users", f"{total_users:,}")
    c3.metric("Catalog problems", f"{total_problems:,}")

    st.markdown("### How it works")
    a, b, c = st.columns(3)
    a.markdown("**1. Choose a mode**\n\nUse a dataset sample or your handle.")
    b.markdown("**2. Analyze practice**\n\nReview solved problems and topic patterns.")
    c.markdown("**3. Find your next task**\n\nExplore ranked unsolved problems.")


# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------
elif page == "Recommendations":
    st.markdown('<div class="section-title">🎯 Your recommendations</div>',
                unsafe_allow_html=True)

    mode_default = st.session_state.get("mode", "Demo")
    mode = st.radio(
        "Recommendation mode",
        ["Demo", "Live Codeforces Handle"],
        index=0 if mode_default == "Demo" else 1,
        horizontal=True,
    )

    rating_filter = st.slider(
        "Difficulty range (Codeforces rating)",
        min_value=800,
        max_value=3500,
        value=(800, 1800),
        step=100,
    )
    number_to_show = st.select_slider(
        "Number of recommendations",
        options=[5, 10, 15, 20],
        value=10,
    )

    if mode == "Demo":
        user_counts = interactions.groupby("user").size()
        eligible = user_counts[user_counts >= 5].index.tolist()
        if not eligible:
            eligible = user_counts.index.tolist()

        if not eligible:
            st.warning("No sample users were found in the dataset.")
            st.stop()

        eligible = sorted(
            eligible, key=lambda u: user_counts.get(u, 0), reverse=True
        )
        selected_user = st.selectbox(
            "Choose a sample dataset user",
            eligible,
            format_func=lambda u: f"Sample user {u}",
        )

        user_data = interactions[interactions["user"] == str(selected_user)]
        solved = set(
            user_data.loc[user_data["solved"] > 0, "join_id"].astype(str)
        )
        attempted = set(user_data["join_id"].astype(str))
        weakness = dataset_topic_weakness(user_data)

        solved_rows = user_data[user_data["solved"] > 0]
        avg_rating = solved_rows["rating"].dropna().mean()
        target = float(avg_rating) + 100 if pd.notna(avg_rating) else 1000

        candidate_catalog = catalog.copy()
        if candidate_catalog.empty:
            st.error("The problems catalog is empty.")
            st.stop()

        if "name" not in candidate_catalog:
            candidate_catalog["name"] = candidate_catalog["problem_id"].astype(str)

        candidate_catalog["rating"] = pd.to_numeric(
            candidate_catalog["rating"], errors="coerce"
        )
        candidate_catalog = candidate_catalog[
            candidate_catalog["rating"].isna()
            | candidate_catalog["rating"].between(*rating_filter)
        ]

        popularity = interactions.groupby("join_id").size().to_dict()
        candidate_catalog["solved_count"] = candidate_catalog["join_id"].map(
            popularity
        ).fillna(0)

        results = rank_problems(
            candidate_catalog, attempted, solved, weakness, target
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Sample user solved", f"{len(solved):,}")
        m2.metric("Recorded interactions", f"{len(user_data):,}")
        m3.metric("Topics observed", f"{len(weakness):,}")

        st.info(
            "Demo mode: these results use historical dataset interactions. "
            "They are not recommendations for a real live account."
        )
        render_recommendations(results, number_to_show)

        st.markdown('<div class="section-title">📚 Topic analysis</div>',
                    unsafe_allow_html=True)
        if weakness:
            topic_df = pd.DataFrame(
                sorted(weakness.items(), key=lambda x: x[1], reverse=True)[:12],
                columns=["Topic", "Weakness score"],
            ).set_index("Topic")
            st.bar_chart(topic_df)
        else:
            st.write("There is not enough topic history to estimate weaknesses.")

    else:
        handle = st.text_input(
            "Enter your Codeforces handle",
            value=st.session_state.get("handle", ""),
            placeholder="e.g. tourist",
        )

        if st.button("Analyze Handle", type="primary"):
            if not handle.strip():
                st.warning("Please enter a Codeforces handle.")
            else:
                try:
                    with st.spinner(
                        "Fetching your Codeforces profile and submissions..."
                    ):
                        profile, submissions = fetch_live_profile(handle.strip())
                    st.session_state["live_profile"] = profile
                    st.session_state["live_submissions"] = submissions
                    st.session_state["handle"] = handle.strip()
                    st.success("Profile loaded successfully.")
                except Exception as exc:
                    st.error(f"Could not load this handle: {exc}")

        profile = st.session_state.get("live_profile")
        submissions = st.session_state.get("live_submissions")

        if profile and submissions is not None:
            solved = set()
            attempted = set()
            for sub in submissions:
                problem = sub.get("problem", {})
                contest = problem.get("contestId")
                index = problem.get("index", "")
                if contest is None:
                    continue
                pid = clean_id(f"{contest}-{index}")
                attempted.add(pid)
                if sub.get("verdict") == "OK":
                    solved.add(pid)

            weakness = live_topic_weakness(submissions)
            ok_subs = [
                s for s in submissions if s.get("verdict") == "OK"
            ]
            solved_ratings = [
                s["problem"]["rating"]
                for s in ok_subs
                if s.get("problem", {}).get("rating") is not None
            ]
            avg_rating = (
                sum(solved_ratings) / len(solved_ratings)
                if solved_ratings else 1000
            )
            target = avg_rating + 100

            p1, p2, p3 = st.columns(3)
            p1.metric("Current rating", profile.get("rating", "Unrated"))
            p2.metric("Max rating", profile.get("maxRating", "—"))
            p3.metric("Solved in fetched history", f"{len(solved):,}")

            st.caption(
                f"Handle: {profile.get('handle', handle)} · "
                f"Fetched {len(submissions):,} submissions at most. "
                "Older history may not be included."
            )

            try:
                with st.spinner("Loading the Codeforces problem catalog..."):
                    live_catalog = get_live_problems()

                live_catalog = live_catalog[
                    live_catalog["rating"].isna()
                    | live_catalog["rating"].between(*rating_filter)
                ]

                results = rank_problems(
                    live_catalog, attempted, solved, weakness, target
                )
                render_recommendations(results, number_to_show)

                st.markdown(
                    '<div class="section-title">📚 Your topic analysis</div>',
                    unsafe_allow_html=True,
                )
                if weakness:
                    topic_df = pd.DataFrame(
                        sorted(
                            weakness.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:12],
                        columns=["Topic", "Weakness score"],
                    ).set_index("Topic")
                    st.bar_chart(topic_df)
                else:
                    st.write(
                        "Not enough repeated topic attempts to estimate "
                        "topic weaknesses reliably."
                    )
            except Exception as exc:
                st.error(f"Could not build recommendations: {exc}")
        else:
            st.info(
                "Enter your handle and click Analyze Handle to get started. "
                "You can use demo mode at any time without an account."
            )


# --------------------------------------------------
# TOPIC ANALYSIS PAGE
# --------------------------------------------------
elif page == "Topic Analysis":
    st.markdown('<div class="section-title">📚 Explore dataset topics</div>',
                unsafe_allow_html=True)
    all_tags = [
        tag
        for tags in interactions["tag_list"]
        for tag in tags
    ]
    if all_tags:
        counts = pd.Series(all_tags).value_counts().head(20)
        st.bar_chart(counts)
        st.caption(
            "These counts show tag frequency in the dataset, not your personal skill."
        )
    else:
        st.info("No topic tags were found in the interaction dataset.")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About":
    st.markdown('<div class="section-title">ℹ️ About this project</div>',
                unsafe_allow_html=True)
    st.write(
        "Codeforces Problem Coach is a prototype that helps competitive "
        "programmers discover practice problems from their topic history "
        "and approximate difficulty level."
    )
    st.markdown("#### Two ways to use the app")
    st.write("- **Demo mode:** uses a sample user from the supplied dataset.")
    st.write("- **Live mode:** uses a public Codeforces handle and public API.")
    st.markdown("#### Important limitations")
    st.write(
        "- The current ranking is a heuristic, not the trained Gradient "
        "Boosting model."
    )
    st.write(
        "- Live analysis is limited to the most recent 10,000 submissions "
        "returned by the Codeforces API."
    )
    st.write(
        "- Ranking scores are relative scores and should not be interpreted "
        "as calibrated probabilities or guaranteed solve success."
    )
    st.write(
        "- Codeforces availability and API rate limits can affect live mode."
    )
