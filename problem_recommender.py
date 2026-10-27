#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q lightgbm joblib pandas numpy scikit-learn')


# In[2]:


from google.colab import files

uploaded = files.upload()


# In[3]:


import os

print(os.listdir())


# In[4]:


import zipfile
import os

zip_path = "Codeforces User–Problem Interaction Dataset for Pe.zip"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("codeforces_dataset")

print("Dataset extracted successfully!")


# In[5]:


for root, folders, files_list in os.walk("codeforces_dataset"):
    for file in files_list:
        print(os.path.join(root, file))


# In[6]:


import pandas as pd

base = "codeforces_dataset/Codeforces User–Problem Interaction Dataset for Pe"

interactions = pd.read_csv(f"{base}/processed_data.csv")
problems = pd.read_csv(f"{base}/problems.csv")
users = pd.read_csv(f"{base}/user_features.csv")

print("Interactions shape:", interactions.shape)
print("Problems shape:", problems.shape)
print("Users shape:", users.shape)


# In[7]:


print("INTERACTIONS COLUMNS:")
print(interactions.columns.tolist())

print("\nPROBLEMS COLUMNS:")
print(problems.columns.tolist())

print("\nUSER FEATURES COLUMNS:")
print(users.columns.tolist())


# In[8]:


tag_weakness = pd.read_csv(
    f"{base}/user_tag_weakness.csv"
)

rating_weakness = pd.read_csv(
    f"{base}/user_rating_weakness.csv"
)

print("USER TAG WEAKNESS")
print("Shape:", tag_weakness.shape)
print("Columns:", tag_weakness.columns.tolist())

print("\nUSER RATING WEAKNESS")
print("Shape:", rating_weakness.shape)
print("Columns:", rating_weakness.columns.tolist())


# In[9]:


display(tag_weakness.head())
display(rating_weakness.head())


# In[10]:


print("========== DATASET SUMMARY ==========")

print("Users:", interactions["user"].nunique())
print("Problems:", interactions["problem_id"].nunique())
print("Interactions:", len(interactions))

print("\n========== MISSING VALUES ==========")
print(interactions.isnull().sum())

print("\n========== DUPLICATES ==========")
print(
    "Duplicate user-problem pairs:",
    interactions.duplicated(
        subset=["user", "problem_id"]
    ).sum()
)

print("\n========== SOLVED DISTRIBUTION ==========")
print(interactions["solved"].value_counts())

print("\n========== RATING SUMMARY ==========")
print(interactions["rating"].describe())


# In[11]:


# Make a copy so the original dataset remains unchanged
interactions_clean = interactions.copy()

# Replace missing tags with an empty string
interactions_clean["tags"] = interactions_clean["tags"].fillna("")

# Make sure solved and attempts are numeric
interactions_clean["solved"] = pd.to_numeric(
    interactions_clean["solved"],
    errors="coerce"
)

interactions_clean["attempts"] = pd.to_numeric(
    interactions_clean["attempts"],
    errors="coerce"
)

# Remove rows that became invalid
interactions_clean = interactions_clean.dropna(
    subset=["user", "problem_id", "solved", "attempts", "rating"]
)

print("Original rows :", len(interactions))
print("Clean rows    :", len(interactions_clean))
print("Missing tags  :", interactions_clean["tags"].isna().sum())


# In[12]:


user_stats = interactions_clean.groupby("user").agg(
    user_solved_count=("solved", "sum"),
    user_total_attempts=("attempts", "sum"),
    user_solve_rate=("solved", "mean"),
    user_avg_attempts=("attempts", "mean"),
    user_avg_problem_rating=("rating", "mean"),
    user_max_problem_rating=("rating", "max")
).reset_index()

print("User feature table shape:", user_stats.shape)

display(user_stats.head())


# In[13]:


problem_stats = interactions_clean.groupby("problem_id").agg(
    problem_interactions=("user", "count"),
    problem_solve_rate=("solved", "mean"),
    problem_avg_attempts=("attempts", "mean")
).reset_index()

# Add problem rating and tags
problem_stats = problem_stats.merge(
    problems[["problem_id", "rating", "tags"]],
    on="problem_id",
    how="left"
)

# Handle missing tags
problem_stats["tags"] = problem_stats["tags"].fillna("")

print("Problem feature table shape:", problem_stats.shape)

display(problem_stats.head())


# In[14]:


# Keep the useful topic-level features
topic_features = tag_weakness[
    [
        "user",
        "tag",
        "solve_rate",
        "mastery",
        "skill_score",
        "weakness_score",
        "is_weak"
    ]
].copy()

# Rename columns to make their purpose clear
topic_features = topic_features.rename(columns={
    "solve_rate": "topic_solve_rate",
    "mastery": "topic_mastery",
    "skill_score": "topic_skill_score",
    "weakness_score": "topic_weakness_score",
    "is_weak": "topic_is_weak"
})

print("Topic feature table shape:", topic_features.shape)

display(topic_features.head())


# In[15]:


def calculate_topic_features(user_id, problem_tags):

    if not problem_tags:
        return {
            "topic_weakness_score": 0.0,
            "topic_solve_rate": 0.0,
            "topic_skill_score": 0.0,
            "topic_mastery": 0.0,
            "weak_topic_count": 0
        }

    tags = [
        tag.strip()
        for tag in str(problem_tags).split(",")
        if tag.strip()
    ]

    user_topics = topic_features[
        topic_features["user"] == user_id
    ]

    matched = user_topics[
        user_topics["tag"].isin(tags)
    ]

    if len(matched) == 0:
        return {
            "topic_weakness_score": 0.0,
            "topic_solve_rate": 0.0,
            "topic_skill_score": 0.0,
            "topic_mastery": 0.0,
            "weak_topic_count": 0
        }

    return {
        "topic_weakness_score": matched["topic_weakness_score"].mean(),
        "topic_solve_rate": matched["topic_solve_rate"].mean(),
        "topic_skill_score": matched["topic_skill_score"].mean(),
        "topic_mastery": matched["topic_mastery"].mean(),
        "weak_topic_count": matched["topic_is_weak"].sum()
    }


# In[16]:


test_user = "-1e11"

test_problem = problem_stats.iloc[0]

features = calculate_topic_features(
    test_user,
    test_problem["tags"]
)

print("Problem:", test_problem["problem_id"])
print("Tags:", test_problem["tags"])
print()
print(features)


# In[17]:


rating_features = rating_weakness[
    [
        "user",
        "rating_bucket",
        "solve_rate",
        "expected_rate",
        "performance_ratio",
        "avg_rating",
        "mastery",
        "skill_score",
        "weakness_score",
        "is_weak"
    ]
].copy()

rating_features = rating_features.rename(columns={
    "solve_rate": "rating_solve_rate",
    "expected_rate": "rating_expected_rate",
    "performance_ratio": "rating_performance_ratio",
    "avg_rating": "rating_avg_rating",
    "mastery": "rating_mastery",
    "skill_score": "rating_skill_score",
    "weakness_score": "rating_weakness_score",
    "is_weak": "rating_is_weak"
})

print("Rating feature table shape:", rating_features.shape)

display(rating_features.head())


# In[18]:


def calculate_rating_features(user_id, problem_rating):

    user_ratings = rating_features[
        rating_features["user"] == user_id
    ].copy()

    # If the user has no rating profile
    if len(user_ratings) == 0:
        return {
            "rating_solve_rate": 0.0,
            "rating_performance_ratio": 0.0,
            "rating_weakness_score": 0.0,
            "rating_skill_score": 0.0,
            "rating_is_weak": 0,
            "rating_distance": 0.0
        }

    # Find the closest known rating bucket
    user_ratings["rating_distance"] = (
        user_ratings["rating_bucket"] - problem_rating
    ).abs()

    closest = user_ratings.loc[
        user_ratings["rating_distance"].idxmin()
    ]

    return {
        "rating_solve_rate": closest["rating_solve_rate"],
        "rating_performance_ratio": closest["rating_performance_ratio"],
        "rating_weakness_score": closest["rating_weakness_score"],
        "rating_skill_score": closest["rating_skill_score"],
        "rating_is_weak": closest["rating_is_weak"],
        "rating_distance": closest["rating_distance"]
    }


# In[19]:


test_user = "-1e11"
test_rating = 1500

difficulty_features = calculate_rating_features(
    test_user,
    test_rating
)

print("User:", test_user)
print("Problem rating:", test_rating)
print()
print(difficulty_features)


# In[20]:


training_data = interactions_clean.copy()

# Add user-level features
training_data = training_data.merge(
    user_stats,
    on="user",
    how="left"
)

# Add problem-level features
training_data = training_data.merge(
    problem_stats[
        [
            "problem_id",
            "problem_interactions",
            "problem_solve_rate",
            "problem_avg_attempts"
        ]
    ],
    on="problem_id",
    how="left"
)

print("Training data shape:", training_data.shape)

display(training_data.head())


# In[21]:


def add_topic_features(row):

    user_id = row["user"]
    problem_tags = row["tags"]

    if not problem_tags:
        return pd.Series({
            "topic_weakness_score": 0.0,
            "topic_solve_rate": 0.0,
            "topic_skill_score": 0.0,
            "topic_mastery": 0.0,
            "weak_topic_count": 0
        })

    tags = [
        tag.strip()
        for tag in str(problem_tags).split(",")
        if tag.strip()
    ]

    matched = topic_features[
        (topic_features["user"] == user_id) &
        (topic_features["tag"].isin(tags))
    ]

    if len(matched) == 0:
        return pd.Series({
            "topic_weakness_score": 0.0,
            "topic_solve_rate": 0.0,
            "topic_skill_score": 0.0,
            "topic_mastery": 0.0,
            "weak_topic_count": 0
        })

    return pd.Series({
        "topic_weakness_score": matched["topic_weakness_score"].mean(),
        "topic_solve_rate": matched["topic_solve_rate"].mean(),
        "topic_skill_score": matched["topic_skill_score"].mean(),
        "topic_mastery": matched["topic_mastery"].mean(),
        "weak_topic_count": matched["topic_is_weak"].sum()
    })


# In[22]:


test_topic_features = training_data.head(10).apply(
    add_topic_features,
    axis=1
)

display(test_topic_features)


# In[23]:


# Split problem tags into individual rows
problem_topic_map = training_data[
    ["user", "problem_id", "tags"]
].copy()

problem_topic_map["tag"] = (
    problem_topic_map["tags"]
    .fillna("")
    .str.split(",")
)

problem_topic_map = problem_topic_map.explode("tag")

problem_topic_map["tag"] = (
    problem_topic_map["tag"]
    .str.strip()
)

# Remove empty tags
problem_topic_map = problem_topic_map[
    problem_topic_map["tag"] != ""
]

print("Expanded user-problem-topic rows:",
      len(problem_topic_map))


# In[24]:


# Merge each user/problem tag with that user's topic profile
topic_merged = problem_topic_map.merge(
    topic_features,
    on=["user", "tag"],
    how="left"
)

print("Merged topic rows:", len(topic_merged))
display(topic_merged.head())


# In[25]:


topic_aggregated = topic_merged.groupby(
    ["user", "problem_id"]
).agg(
    topic_weakness_score=(
        "topic_weakness_score", "mean"
    ),
    topic_solve_rate=(
        "topic_solve_rate", "mean"
    ),
    topic_skill_score=(
        "topic_skill_score", "mean"
    ),
    topic_mastery=(
        "topic_mastery", "mean"
    ),
    weak_topic_count=(
        "topic_is_weak", "sum"
    )
).reset_index()

print(
    "Aggregated topic feature rows:",
    len(topic_aggregated)
)

display(topic_aggregated.head())


# In[26]:


training_data = training_data.merge(
    topic_aggregated,
    on=["user", "problem_id"],
    how="left"
)

print("Training data shape:", training_data.shape)

print("\nMissing topic features:")
print(
    training_data[
        [
            "topic_weakness_score",
            "topic_solve_rate",
            "topic_skill_score",
            "topic_mastery",
            "weak_topic_count"
        ]
    ].isna().sum()
)


# In[27]:


topic_columns = [
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count"
]

training_data[topic_columns] = training_data[topic_columns].fillna(0)

print("Missing topic features after filling:")
print(training_data[topic_columns].isna().sum())


# In[28]:


def get_rating_features_for_row(row):

    user_id = row["user"]
    problem_rating = row["rating"]

    user_ratings = rating_features[
        rating_features["user"] == user_id
    ]

    if len(user_ratings) == 0:
        return pd.Series({
            "rating_solve_rate": 0.0,
            "rating_performance_ratio": 0.0,
            "rating_weakness_score": 0.0,
            "rating_skill_score": 0.0,
            "rating_is_weak": 0,
            "rating_distance": 0.0
        })

    distances = (
        user_ratings["rating_bucket"] - problem_rating
    ).abs()

    closest_idx = distances.idxmin()
    closest = user_ratings.loc[closest_idx]

    return pd.Series({
        "rating_solve_rate": closest["rating_solve_rate"],
        "rating_performance_ratio": closest["rating_performance_ratio"],
        "rating_weakness_score": closest["rating_weakness_score"],
        "rating_skill_score": closest["rating_skill_score"],
        "rating_is_weak": closest["rating_is_weak"],
        "rating_distance": distances.loc[closest_idx]
    })


# In[29]:


test_rating_features = training_data.head(10).apply(
    get_rating_features_for_row,
    axis=1
)

display(test_rating_features)


# In[30]:


# Create user-problem rating mapping
rating_map = training_data[
    ["user", "problem_id", "rating"]
].copy()

rating_map["problem_rating"] = rating_map["rating"]

# IMPORTANT:
# merge_asof requires the actual merge key to be sorted
rating_map = rating_map.sort_values(
    ["problem_rating", "user"]
).reset_index(drop=True)

rating_features_sorted = rating_features.sort_values(
    ["rating_bucket", "user"]
).reset_index(drop=True)

# Find nearest rating bucket for each user/problem
rating_matched = pd.merge_asof(
    rating_map,
    rating_features_sorted,
    left_on="problem_rating",
    right_on="rating_bucket",
    by="user",
    direction="nearest"
)

print("Rating matched rows:", len(rating_matched))

display(rating_matched.head(10))


# In[31]:


rating_columns = [
    "user",
    "problem_id",
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]

rating_features_for_merge = rating_matched[rating_columns].copy()

training_data = training_data.merge(
    rating_features_for_merge,
    on=["user", "problem_id"],
    how="left"
)

print("Final training data shape:", training_data.shape)

print("\nMissing values:")
print(
    training_data.isna().sum()
)


# In[32]:


feature_columns = [
    # User performance
    "user_solved_count",
    "user_total_attempts",
    "user_solve_rate",
    "user_avg_attempts",
    "user_avg_problem_rating",
    "user_max_problem_rating",

    # Problem characteristics
    "rating",
    "problem_interactions",
    "problem_solve_rate",
    "problem_avg_attempts",

    # Topic personalization
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count",

    # Difficulty personalization
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]

X = training_data[feature_columns]
y = training_data["solved"]

print("Number of features:", len(feature_columns))
print("X shape:", X.shape)
print("y shape:", y.shape)

display(X.head())


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# In[34]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

print("Model training completed!")


# In[35]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# Predictions
y_pred = model.predict(X_test)

# Probability of class 1 (solved)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[36]:


import pandas as pd

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("TOP 15 FEATURES")
display(feature_importance.head(15))


# In[37]:


import matplotlib.pyplot as plt

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features Used by the Recommendation Model")
plt.show()


# In[38]:


test_user = "-1e11"

print("Testing recommendation system for:", test_user)

# Problems already attempted by this user
attempted_problems = set(
    training_data.loc[
        training_data["user"] == test_user,
        "problem_id"
    ]
)

print("Problems already attempted:", len(attempted_problems))


# In[39]:


print([x for x in globals() if 'problem' in x.lower()])


# In[40]:


candidate_problems = problems[
    ~problems["problem_id"].isin(attempted_problems)
].copy()

print("Total problems:", len(problems))
print("Already attempted:", len(attempted_problems))
print("Candidate problems:", len(candidate_problems))

display(
    candidate_problems[
        ["problem_id", "rating", "tags"]
    ].head(10)
)


# In[41]:


print([x for x in globals() if 'user' in x.lower()])


# In[42]:


print("Shape:", user_stats.shape)
print("Columns:")
print(user_stats.columns.tolist())

display(user_stats.head())


# In[43]:


user_feature_columns = [
    "user_solved_count",
    "user_total_attempts",
    "user_solve_rate",
    "user_avg_attempts",
    "user_avg_problem_rating",
    "user_max_problem_rating"
]

# Get the selected user's profile
user_profile = user_stats[
    user_stats["user"] == test_user
].iloc[0]

# Copy candidate problems
candidate_data = candidate_problems.copy()

# Add this user's features to every candidate problem
for col in user_feature_columns:
    candidate_data[col] = user_profile[col]

print("Candidate data shape:", candidate_data.shape)

display(
    candidate_data[
        [
            "problem_id",
            "rating",
            "tags"
        ] + user_feature_columns
    ].head(10)
)


# In[44]:


print([
    x for x in globals()
    if "tag" in x.lower() or "topic" in x.lower()
])


# In[45]:


print("Shape:", topic_features.shape)
print("Columns:")
print(topic_features.columns.tolist())

display(topic_features.head())


# In[46]:


# Copy candidate problems
candidate_topics = candidate_data[
    ["problem_id", "tags"]
].copy()

# Split tags into individual topics
candidate_topics["tag"] = (
    candidate_topics["tags"]
    .fillna("")
    .str.split(",")
)

# One row per problem-topic
candidate_topics = candidate_topics.explode("tag")

candidate_topics["tag"] = (
    candidate_topics["tag"]
    .str.strip()
)

# Get only the selected user's topic history
user_topic_features = topic_features[
    topic_features["user"] == test_user
].copy()

print("User topic records:", len(user_topic_features))

display(user_topic_features.head())


# In[47]:


candidate_topics = candidate_topics.merge(
    user_topic_features[
        [
            "tag",
            "topic_solve_rate",
            "topic_mastery",
            "topic_skill_score",
            "topic_weakness_score",
            "topic_is_weak"
        ]
    ],
    on="tag",
    how="left"
)

print("Candidate-topic rows:", len(candidate_topics))

display(candidate_topics.head(10))


# In[48]:


topic_aggregated_candidates = (
    candidate_topics
    .groupby("problem_id")
    .agg(
        topic_weakness_score=("topic_weakness_score", "mean"),
        topic_solve_rate=("topic_solve_rate", "mean"),
        topic_skill_score=("topic_skill_score", "mean"),
        topic_mastery=("topic_mastery", "mean"),
        weak_topic_count=("topic_is_weak", "sum")
    )
    .reset_index()
)

print(
    "Aggregated topic feature rows:",
    len(topic_aggregated_candidates)
)

display(topic_aggregated_candidates.head(10))


# In[49]:


candidate_data = candidate_data.merge(
    topic_aggregated_candidates,
    on="problem_id",
    how="left"
)

topic_feature_names = [
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count"
]

candidate_data[topic_feature_names] = (
    candidate_data[topic_feature_names]
    .fillna(0)
)

print("Candidate data shape:", candidate_data.shape)

display(
    candidate_data[
        [
            "problem_id",
            "rating",
            "tags"
        ] + topic_feature_names
    ].head(10)
)


# In[50]:


print([
    x for x in globals()
    if "rating" in x.lower()
])


# In[51]:


print("Shape:", rating_features.shape)
print("Columns:")
print(rating_features.columns.tolist())

display(rating_features.head())


# In[52]:


# ============================================================
# FIX — ADD PROBLEM FEATURES
# ============================================================

print("problems columns:")
print(problems.columns.tolist())

print("\nproblem_stats columns:")
print(problem_stats.columns.tolist())

# Keep the problem-level features used during training
problem_feature_columns = [
    "problem_id",
    "problem_interactions",
    "problem_solve_rate",
    "problem_avg_attempts"
]

problem_stats_small = problem_stats[
    problem_feature_columns
].copy()

# Merge problem statistics into candidate problems
candidate_data = candidate_data.merge(
    problem_stats_small,
    on="problem_id",
    how="left"
)

# Handle any missing values
candidate_data[
    [
        "problem_interactions",
        "problem_solve_rate",
        "problem_avg_attempts"
    ]
] = candidate_data[
    [
        "problem_interactions",
        "problem_solve_rate",
        "problem_avg_attempts"
    ]
].fillna(0)

print("\nCandidate data shape:", candidate_data.shape)

display(
    candidate_data[
        [
            "problem_id",
            "rating",
            "problem_interactions",
            "problem_solve_rate",
            "problem_avg_attempts"
        ]
    ].head(10)
)


# In[53]:


# ============================================================
# STEP 7 — BUILD EXACT 23-FEATURE MATRIX
# ============================================================

import numpy as np

# List of rating feature names (excluding 'problem_id')
rating_feature_names_list = [
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]

# --- Add Difficulty Personalization features to candidate_data ---
# Get the test_user's rating features from the rating_features DataFrame
user_specific_rating_features = rating_features[rating_features["user"] == test_user].copy()

# Create a temporary DataFrame from candidate_data containing problem_id and problem rating
candidate_problem_ratings_for_merge = candidate_data[['problem_id', 'rating']].copy()
candidate_problem_ratings_for_merge = candidate_problem_ratings_for_merge.rename(columns={'rating': 'problem_rating'})

if not user_specific_rating_features.empty:
    # Prepare both DataFrames by sorting them on their respective rating columns
    # This is a prerequisite for pd.merge_asof
    candidate_problem_ratings_for_merge_sorted = candidate_problem_ratings_for_merge.sort_values(
        'problem_rating'
    ).reset_index(drop=True)

    user_specific_rating_features_sorted = user_specific_rating_features.sort_values(
        'rating_bucket'
    ).reset_index(drop=True)

    # Perform merge_asof to find the closest rating bucket for each candidate problem
    # For a single user, the 'by' parameter can be omitted or a dummy column can be used.
    # We will just merge on problem_rating and rating_bucket for this single user.
    merged_rating_info = pd.merge_asof(
        candidate_problem_ratings_for_merge_sorted,
        user_specific_rating_features_sorted,
        left_on='problem_rating',
        right_on='rating_bucket',
        direction='nearest'
    )

    # Define the specific rating columns to extract and merge into candidate_data
    rating_cols_to_extract = ["problem_id"] + rating_feature_names_list

    rating_features_for_candidates_df = merged_rating_info[rating_cols_to_extract]

    # --- FIX: Drop existing rating feature columns from candidate_data to avoid MergeError ---
    # Identify columns in candidate_data that would conflict with the incoming rating features
    conflicting_cols = [col for col in rating_feature_names_list if col in candidate_data.columns]
    if conflicting_cols:
        candidate_data = candidate_data.drop(columns=conflicting_cols)
    # --- End FIX ---

    # Merge these new rating features into the main candidate_data DataFrame
    candidate_data = candidate_data.merge(
        rating_features_for_candidates_df,
        on="problem_id",
        how="left"
    )

# Fill any NaN values in the newly added rating features with 0.0
# This handles cases where a user might not have a rating profile at all, or edge cases from merge_asof
for col in rating_feature_names_list:
    if col not in candidate_data.columns: # Add column if it doesn't exist
        candidate_data[col] = 0.0
    candidate_data[col] = candidate_data[col].fillna(0.0)

# --- End of Difficulty Personalization features addition ---

feature_columns = [
    # User features — 6
    "user_solved_count",
    "user_total_attempts",
    "user_solve_rate",
    "user_avg_attempts",
    "user_avg_problem_rating",
    "user_max_problem_rating",

    # Problem features — 4
    "rating",
    "problem_interactions",
    "problem_solve_rate",
    "problem_avg_attempts",

    # Topic features — 5
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count",

    # Rating features — 8
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]

X_candidates = candidate_data[feature_columns].copy()

# Clean numerical values
X_candidates = X_candidates.replace(
    [np.inf, -np.inf],
    np.nan
)

X_candidates = X_candidates.fillna(0)

print("=" * 60)
print("FINAL CANDIDATE FEATURE MATRIX")
print("=" * 60)

print("Number of features:", X_candidates.shape[1])
print("X shape:", X_candidates.shape)
print("Missing values:", X_candidates.isna().sum().sum())

display(X_candidates.head())


# In[54]:


# ============================================================
# STEP 29 — PREDICT RECOMMENDATION SCORES
# ============================================================

print("=" * 70)
print("GENERATING RECOMMENDATION SCORES")
print("=" * 70)

# Check that the trained model exists
print("Model type:", type(model).__name__)

# Predict probability of solving each candidate problem
candidate_data["recommendation_score"] = (
    model.predict_proba(X_candidates)[:, 1]
)

print("Scores generated:", len(candidate_data))
print(
    "Score range:",
    candidate_data["recommendation_score"].min(),
    "to",
    candidate_data["recommendation_score"].max()
)

display(
    candidate_data[
        ["problem_id", "rating", "tags", "recommendation_score"]
    ].head(10)
)


# In[55]:


# ============================================================
# STEP 30 — RANK RECOMMENDATIONS
# ============================================================

recommendations = candidate_data.sort_values(
    "recommendation_score",
    ascending=False
).reset_index(drop=True)

print("=" * 70)
print("RANKING COMPLETED")
print("=" * 70)

print("Total candidates ranked:", len(recommendations))

display(
    recommendations[
        [
            "problem_id",
            "rating",
            "tags",
            "recommendation_score",
            "topic_weakness_score",
            "weak_topic_count"
        ]
    ].head(20)
)


# In[56]:


# ============================================================
# STEP 31 — FINAL TOP 10 RECOMMENDATIONS
# ============================================================

top_10 = recommendations.head(10).copy()

print("=" * 70)
print("PERSONALIZED CODEFORCES RECOMMENDATIONS")
print("=" * 70)

print("User:", test_user)
print()

for i, (_, row) in enumerate(top_10.iterrows(), 1):

    print(f"#{i}")
    print(f"Problem ID          : {row['problem_id']}")
    print(f"Rating              : {row['rating']}")
    print(f"Tags                : {row['tags']}")
    print(
        f"Recommendation Score: "
        f"{row['recommendation_score']:.4f}"
    )
    print(
        f"Topic Weakness      : "
        f"{row['topic_weakness_score']:.4f}"
    )
    print(
        f"Weak Topics         : "
        f"{int(row['weak_topic_count'])}"
    )

    print("-" * 70)


# ## Person - 2

# In[57]:


# PERSON 2 — ROUND 2 — M2
# STEP 1: CHECK FINAL ML DATA

print("=" * 60)
print("CHECKING FINAL ML DATA")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nNumber of features:", X.shape[1])

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# In[58]:


#Create train/test data
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# In[59]:


# STEP 3: LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

print("=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")


# In[60]:


# STEP 4: LOGISTIC REGRESSION EVALUATION

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

y_pred_lr = logistic_model.predict(X_test)
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
pr_auc_lr = average_precision_score(y_test, y_prob_lr)

print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy_lr:.4f}")
print(f"Precision : {precision_lr:.4f}")
print(f"Recall    : {recall_lr:.4f}")
print(f"F1 Score  : {f1_lr:.4f}")
print(f"ROC-AUC   : {roc_auc_lr:.4f}")
print(f"PR-AUC    : {pr_auc_lr:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))


# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


#Train model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print("Random Forest training completed.")


# In[63]:


#Make predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Predictions completed.")


# In[64]:


#Model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision : {precision_score(y_test, rf_pred):.4f}")
print(f"Recall    : {recall_score(y_test, rf_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, rf_prob):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, rf_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# In[65]:


#Feature importance
import pandas as pd

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("TOP 15 RANDOM FOREST FEATURES")
print("=" * 60)
print(feature_importance.head(15))


# In[66]:


#Plot top features
import matplotlib.pyplot as plt

top_features = feature_importance.head(15).sort_values(
    by="importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features Used by Random Forest")

plt.tight_layout()
plt.show()


# In[67]:


from sklearn.ensemble import GradientBoostingClassifier


# In[68]:


#Train gradient boosting
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

print("Gradient Boosting training completed.")


# In[69]:


#Predict
gb_pred = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)[:, 1]

print("Predictions completed.")


# In[70]:


#Evaluate
print("=" * 60)
print("GRADIENT BOOSTING RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy_score(y_test, gb_pred):.4f}")
print(f"Precision : {precision_score(y_test, gb_pred):.4f}")
print(f"Recall    : {recall_score(y_test, gb_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, gb_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, gb_prob):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, gb_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, gb_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, gb_pred))


# In[71]:


gb_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": gb_model.feature_importances_
})

gb_importance = gb_importance.sort_values(
    by="importance",
    ascending=False
)

print("TOP 15 GRADIENT BOOSTING FEATURES")
print("=" * 60)
print(gb_importance.head(15))


# In[72]:


model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [0.7285, 0.8523, 0.9550],
    "Precision": [0.9818, 0.9813, 0.9570],
    "Recall": [0.7285, 0.8614, 0.9976],
    "F1": [0.8364, 0.9175, 0.9769],
    "ROC_AUC": [0.8045, 0.8704, 0.8738],
    "PR_AUC": [0.9869, 0.9921, 0.9924]
})

print(model_results)


# In[73]:


# ============================================================
# PERSON 2 — STEP 9
# MODEL COMPARISON SUMMARY
# ============================================================

print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

display(
    model_results.style
    .format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "ROC_AUC": "{:.4f}",
        "PR_AUC": "{:.4f}"
    })
)


# In[74]:


# ============================================================
# PERSON 2 — STEP 10
# ROC-AUC vs PR-AUC
# ============================================================

auc_results = model_results[
    ["Model", "ROC_AUC", "PR_AUC"]
].copy()

x = np.arange(len(auc_results))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    auc_results["ROC_AUC"],
    width,
    label="ROC-AUC"
)

plt.bar(
    x + width / 2,
    auc_results["PR_AUC"],
    width,
    label="PR-AUC"
)

plt.xticks(x, auc_results["Model"])
plt.ylim(0, 1.05)

plt.xlabel("Model")
plt.ylabel("AUC Score")
plt.title("ROC-AUC and PR-AUC Comparison")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()


# In[75]:


# ============================================================
# PERSON 2 — STEP 11
# BEST RECORDED METRIC VALUES
# ============================================================

print("=" * 70)
print("HIGHEST RECORDED VALUES")
print("=" * 70)

for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:

    idx = model_results[metric].idxmax()

    print(
        f"{metric:12s}: "
        f"{model_results.loc[idx, 'Model']} "
        f"({model_results.loc[idx, metric]:.4f})"
    )


# In[76]:


# ============================================================
# PERSON 2 — STEP 12
# ROC CURVE COMPARISON
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ------------------------------------------------------------
# Get probability predictions
# ------------------------------------------------------------

# Logistic Regression
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

# Random Forest
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Gradient Boosting
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# Calculate ROC curves
# ------------------------------------------------------------

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)


# ------------------------------------------------------------
# Calculate AUC
# ------------------------------------------------------------

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_gb = auc(fpr_gb, tpr_gb)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    fpr_lr,
    tpr_lr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {auc_lr:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2,
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

plt.plot(
    fpr_gb,
    tpr_gb,
    linewidth=2,
    label=f"Gradient Boosting (AUC = {auc_gb:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# Print actual AUC values
# ------------------------------------------------------------

print("=" * 60)
print("ROC-AUC VALUES")
print("=" * 60)

print(f"Logistic Regression : {auc_lr:.4f}")
print(f"Random Forest       : {auc_rf:.4f}")
print(f"Gradient Boosting   : {auc_gb:.4f}")


# In[77]:


# ============================================================
# PERSON 2 — STEP 13
# PRECISION-RECALL CURVE COMPARISON
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# ------------------------------------------------------------
# Probability predictions
# ------------------------------------------------------------

# Logistic Regression
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

# Random Forest
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Gradient Boosting
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# Calculate Precision-Recall curves
# ------------------------------------------------------------

precision_lr, recall_lr, _ = precision_recall_curve(
    y_test, y_prob_lr
)

precision_rf, recall_rf, _ = precision_recall_curve(
    y_test, y_prob_rf
)

precision_gb, recall_gb, _ = precision_recall_curve(
    y_test, y_prob_gb
)


# ------------------------------------------------------------
# Calculate PR-AUC
# ------------------------------------------------------------

pr_auc_lr = average_precision_score(
    y_test, y_prob_lr
)

pr_auc_rf = average_precision_score(
    y_test, y_prob_rf
)

pr_auc_gb = average_precision_score(
    y_test, y_prob_gb
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    recall_lr,
    precision_lr,
    linewidth=2,
    label=f"Logistic Regression (PR-AUC = {pr_auc_lr:.4f})"
)

plt.plot(
    recall_rf,
    precision_rf,
    linewidth=2,
    label=f"Random Forest (PR-AUC = {pr_auc_rf:.4f})"
)

plt.plot(
    recall_gb,
    precision_gb,
    linewidth=2,
    label=f"Gradient Boosting (PR-AUC = {pr_auc_gb:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# Print actual PR-AUC values
# ------------------------------------------------------------

print("=" * 60)
print("PRECISION-RECALL AUC VALUES")
print("=" * 60)

print(f"Logistic Regression : {pr_auc_lr:.4f}")
print(f"Random Forest       : {pr_auc_rf:.4f}")
print(f"Gradient Boosting   : {pr_auc_gb:.4f}")


# In[78]:


# ============================================================
# PERSON 2 — STEP 14
# CONFUSION MATRIX COMPARISON
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

y_pred_lr = logistic_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)


# ------------------------------------------------------------
# Calculate confusion matrices
# ------------------------------------------------------------

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_gb = confusion_matrix(y_test, y_pred_gb)


# ------------------------------------------------------------
# Print confusion matrices
# ------------------------------------------------------------

print("=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

print("\nLogistic Regression:")
print(cm_lr)

print("\nRandom Forest:")
print(cm_rf)

print("\nGradient Boosting:")
print(cm_gb)


# ------------------------------------------------------------
# Plot Logistic Regression
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Unsolved (0)", "Solved (1)"]
)

disp.plot()
plt.title("Logistic Regression — Confusion Matrix")
plt.show()


# ------------------------------------------------------------
# Plot Random Forest
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Unsolved (0)", "Solved (1)"]
)

disp.plot()
plt.title("Random Forest — Confusion Matrix")
plt.show()


# ------------------------------------------------------------
# Plot Gradient Boosting
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_gb,
    display_labels=["Unsolved (0)", "Solved (1)"]
)

disp.plot()
plt.title("Gradient Boosting — Confusion Matrix")
plt.show()


# In[79]:


# ============================================================
# PERSON 2 — STEP 15
# FEATURE IMPORTANCE COMPARISON
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Get feature names
# ------------------------------------------------------------

feature_names = X_train.columns.tolist()

# ------------------------------------------------------------
# Random Forest feature importance
# ------------------------------------------------------------

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

# ------------------------------------------------------------
# Gradient Boosting feature importance
# ------------------------------------------------------------

gb_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": gb_model.feature_importances_
})

gb_importance = gb_importance.sort_values(
    by="Importance",
    ascending=False
)


# ------------------------------------------------------------
# Print top 15 features
# ------------------------------------------------------------

print("=" * 70)
print("RANDOM FOREST — TOP 15 FEATURES")
print("=" * 70)

print(rf_importance.head(15).to_string(index=False))


print("\n")
print("=" * 70)
print("GRADIENT BOOSTING — TOP 15 FEATURES")
print("=" * 70)

print(gb_importance.head(15).to_string(index=False))


# ------------------------------------------------------------
# Plot Random Forest
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    rf_importance.head(15)["Feature"][::-1],
    rf_importance.head(15)["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest — Top 15 Feature Importances")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot Gradient Boosting
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    gb_importance.head(15)["Feature"][::-1],
    gb_importance.head(15)["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Gradient Boosting — Top 15 Feature Importances")

plt.tight_layout()
plt.show()


# In[80]:


# ============================================================
# PERSON 2 — STEP 16
# FINAL MODEL COMPARISON
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# Create final comparison table
# ------------------------------------------------------------

final_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Accuracy": [
        0.7285,
        0.8523,
        0.9550
    ],

    "Precision": [
        0.9818,
        0.9813,
        0.9570
    ],

    "Recall": [
        0.7285,
        0.8614,
        0.9976
    ],

    "F1": [
        0.8364,
        0.9175,
        0.9769
    ],

    "ROC_AUC": [
        auc_lr,
        auc_rf,
        auc_gb
    ],

    "PR_AUC": [
        pr_auc_lr,
        pr_auc_rf,
        pr_auc_gb
    ]
})


# ------------------------------------------------------------
# Display rounded results
# ------------------------------------------------------------

print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    final_model_comparison.round(4).to_string(index=False)
)


# In[30]:


# Step 11: Rebuild the complete training dataset
# Corrected version: add the problem rating from `problems`

# Start with the original interactions
training_data = interactions.copy()

print("Initial interaction data:", training_data.shape)

# ---------------------------------------------------------
# 1. Add user-level features
# ---------------------------------------------------------
training_data = training_data.merge(
    user_stats,
    on="user",
    how="left"
)

print("After user features:", training_data.shape)

# ---------------------------------------------------------
# 2. Add problem-level statistics
# ---------------------------------------------------------
training_data = training_data.merge(
    problem_stats,
    on="problem_id",
    how="left"
)

print("After problem statistics:", training_data.shape)

# ---------------------------------------------------------
# 3. Add the original problem rating
# ---------------------------------------------------------
# problem_stats does not contain `rating`,
# so we get it from the problems table.
problem_rating = problems[["problem_id", "rating"]].drop_duplicates(
    subset=["problem_id"]
)

training_data = training_data.merge(
    problem_rating,
    on="problem_id",
    how="left"
)

print("After problem rating:", training_data.shape)

# ---------------------------------------------------------
# 4. Add topic-level features
# ---------------------------------------------------------
training_data = training_data.merge(
    topic_aggregated,
    on=["user", "problem_id"],
    how="left"
)

print("After topic features:", training_data.shape)

# ---------------------------------------------------------
# 5. Create the closest rating bucket
# ---------------------------------------------------------
rating_bucket_values = sorted(
    rating_stats["rating_bucket"].dropna().unique()
)

training_data["rating_bucket"] = training_data["rating"].apply(
    lambda x: min(
        rating_bucket_values,
        key=lambda bucket: abs(bucket - x)
    )
)

# ---------------------------------------------------------
# 6. Add rating-level features
# ---------------------------------------------------------
training_data = training_data.merge(
    rating_stats,
    on=["user", "rating_bucket"],
    how="left"
)

print("After rating features:", training_data.shape)

# ---------------------------------------------------------
# 7. Clean missing/infinite values
# ---------------------------------------------------------
training_data = training_data.replace(
    [np.inf, -np.inf],
    np.nan
)

training_data = training_data.fillna(0)

print("\nFinal training dataset:", training_data.shape)
print("Total missing values:", training_data.isna().sum().sum())


# In[31]:


# Step 12: Create the ML feature matrix (X) and target (y)

# The 23 features used by the original model
X = training_data[feature_columns].copy()

# Target variable: solved
y = training_data["solved"].copy()

# Clean any remaining infinite values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nMissing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# In[19]:


import joblib

joblib.dump(gb_model, "gradient_boosting_model.pkl")

print("Model saved!")


# In[32]:


# Step 13: Split the data into training and testing sets

from sklearn.model_selection import train_test_split

# Use the same 80/20 split
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining solved proportion:", y_train.mean())
print("Testing solved proportion:", y_test.mean())


# In[33]:


# Step 14: Train the Gradient Boosting model

from sklearn.ensemble import GradientBoostingClassifier

# Create the Gradient Boosting classifier
gb_model = GradientBoostingClassifier(
    random_state=42
)

# Train the model
print("Training Gradient Boosting model...")
gb_model.fit(X_train, y_train)

print("Training completed!")
print("Model type:", type(gb_model))


# In[34]:


# Step 15: Verify that the model is fitted and save it

from sklearn.utils.validation import check_is_fitted
import joblib
import os

# Verify the model is actually trained
check_is_fitted(gb_model)

print("Model is FITTED.")

# Save the trained model
model_path = "gradient_boosting_model.pkl"
joblib.dump(gb_model, model_path)

print("Model saved successfully!")
print("File location:", os.path.abspath(model_path))


# In[81]:


test_user = users.iloc[0]["user"]

print("Test User:", test_user)


# ## Person 3

# In[5]:


# ============================================================
# PERSON 3 - SETUP
# Load the existing Codeforces data and feature tables
# ============================================================

import pandas as pd
import numpy as np
import os
import zipfile

# ------------------------------------------------------------
# 1. Define the dataset ZIP path
# ------------------------------------------------------------

zip_path = os.path.join(
    os.getcwd(),
    "Codeforces User–Problem Interaction Dataset for Pe.zip"
)

# ------------------------------------------------------------
# 2. Extract the dataset if it has not already been extracted
# ------------------------------------------------------------

extract_folder = os.path.join(
    os.getcwd(),
    "codeforces_person3_data"
)

if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

print("Dataset folder:", extract_folder)

# ------------------------------------------------------------
# 3. Locate the extracted dataset folder
# ------------------------------------------------------------

dataset_folder = os.path.join(
    extract_folder,
    "Codeforces User–Problem Interaction Dataset for Pe"
)

print("Dataset files:")
for file in os.listdir(dataset_folder):
    print(" -", file)

# ------------------------------------------------------------
# 4. Load the main interaction dataset
# ------------------------------------------------------------

interactions = pd.read_csv(
    os.path.join(dataset_folder, "processed_data.csv")
)

print("\nInteractions shape:", interactions.shape)

# ------------------------------------------------------------
# 5. Load the problem dataset
# ------------------------------------------------------------

problems = pd.read_csv(
    os.path.join(dataset_folder, "problems.csv")
)

print("Problems shape:", problems.shape)

# ------------------------------------------------------------
# 6. Load user-level features
# ------------------------------------------------------------

user_stats = pd.read_csv(
    os.path.join(dataset_folder, "user_features.csv")
)

print("User features shape:", user_stats.shape)

# ------------------------------------------------------------
# 7. Load topic-level weakness features
# ------------------------------------------------------------

topic_weakness = pd.read_csv(
    os.path.join(dataset_folder, "user_tag_weakness.csv")
)

print("Topic weakness shape:", topic_weakness.shape)

# ------------------------------------------------------------
# 8. Load rating-level weakness features
# ------------------------------------------------------------

rating_features = pd.read_csv(
    os.path.join(dataset_folder, "user_rating_weakness.csv")
)

print("Rating features shape:", rating_features.shape)

# ------------------------------------------------------------
# 9. Create the users table
# ------------------------------------------------------------

users = pd.DataFrame({
    "user": interactions["user"].drop_duplicates().values
})

print("Number of users:", len(users))

# ------------------------------------------------------------
# 10. Display the columns so we can verify the data
# ------------------------------------------------------------

print("\nInteraction columns:")
print(interactions.columns.tolist())

print("\nProblem columns:")
print(problems.columns.tolist())

print("\nUser feature columns:")
print(user_stats.columns.tolist())

print("\nTopic feature columns:")
print(topic_weakness.columns.tolist())

print("\nRating feature columns:")
print(rating_features.columns.tolist())


# In[6]:


# ============================================================
# PERSON 3 - STEP 2
# Recreate the feature tables needed for recommendations
# ============================================================

# ------------------------------------------------------------
# 1. Rename user feature columns to match the ML pipeline
# ------------------------------------------------------------

user_stats = user_stats.rename(columns={
    "solved": "user_solved_count",
    "attempts": "user_total_attempts",
    "solve_rate": "user_solve_rate",
    "avg_attempts": "user_avg_attempts",
    "avg_problem_rating": "user_avg_problem_rating",
    "max_problem_rating": "user_max_problem_rating"
})

# Keep only the features used by the recommendation system
user_stats = user_stats[
    [
        "user",
        "user_solved_count",
        "user_total_attempts",
        "user_solve_rate",
        "user_avg_attempts",
        "user_avg_problem_rating",
        "user_max_problem_rating"
    ]
].copy()

print("User feature table:", user_stats.shape)


# ------------------------------------------------------------
# 2. Create problem-level statistics
# ------------------------------------------------------------

problem_stats = (
    interactions
    .groupby("problem_id")
    .agg(
        problem_interactions=("user", "count"),
        problem_solve_rate=("solved", "mean"),
        problem_avg_attempts=("attempts", "mean")
    )
    .reset_index()
)

# Add problem rating and tags from the problems table
problem_stats = problem_stats.merge(
    problems[["problem_id", "rating", "tags"]],
    on="problem_id",
    how="left"
)

print("Problem feature table:", problem_stats.shape)


# ------------------------------------------------------------
# 3. Recreate topic-level user features
# ------------------------------------------------------------

topic_features = topic_weakness.rename(columns={
    "solve_rate": "topic_solve_rate",
    "mastery": "topic_mastery",
    "skill_score": "topic_skill_score",
    "weakness_score": "topic_weakness_score",
    "is_weak": "topic_is_weak"
})

topic_features = topic_features[
    [
        "user",
        "tag",
        "topic_solve_rate",
        "topic_mastery",
        "topic_skill_score",
        "topic_weakness_score",
        "topic_is_weak"
    ]
].copy()

print("Topic feature table:", topic_features.shape)


# ------------------------------------------------------------
# 4. Create user-problem topic features
# ------------------------------------------------------------
# A problem can contain multiple tags.
# We calculate the user's topic weakness across the tags
# belonging to each candidate problem.

# Split problem tags into individual topics
problem_tags = problems[
    ["problem_id", "tags"]
].copy()

problem_tags["tag"] = problem_tags["tags"].fillna("").str.split(",")

problem_tags = problem_tags.explode("tag")

# Remove unnecessary whitespace
problem_tags["tag"] = problem_tags["tag"].str.strip()

# Remove empty tags
problem_tags = problem_tags[
    problem_tags["tag"] != ""
].copy()

# Connect each user with the tags of each problem
user_problem_topics = problem_tags.merge(
    topic_features,
    on="tag",
    how="inner"
)

# Keep only users/problems that actually occur together
user_problem_topics = user_problem_topics[
    [
        "user",
        "problem_id",
        "topic_weakness_score",
        "topic_solve_rate",
        "topic_skill_score",
        "topic_mastery",
        "topic_is_weak"
    ]
]

# Aggregate topic information for each user-problem pair
topic_aggregated = (
    user_problem_topics
    .groupby(["user", "problem_id"])
    .agg(
        topic_weakness_score=("topic_weakness_score", "mean"),
        topic_solve_rate=("topic_solve_rate", "mean"),
        topic_skill_score=("topic_skill_score", "mean"),
        topic_mastery=("topic_mastery", "mean"),
        weak_topic_count=("topic_is_weak", "sum")
    )
    .reset_index()
)

print("Aggregated topic features:", topic_aggregated.shape)


# ------------------------------------------------------------
# 5. Display the recreated feature tables
# ------------------------------------------------------------

print("\nUser features:")
print(user_stats.head())

print("\nProblem features:")
print(problem_stats.head())

print("\nTopic features:")
print(topic_aggregated.head())


# In[7]:


# ============================================================
# PERSON 3 - STEP 3
# Prepare rating-level features
# ============================================================

# Rename the rating feature columns so they match
# the feature names used by the ML model.

rating_stats = rating_features.rename(columns={
    "solve_rate": "rating_solve_rate",
    "expected_rate": "rating_expected_rate",
    "performance_ratio": "rating_performance_ratio",
    "avg_rating": "rating_avg_rating",
    "mastery": "rating_mastery",
    "skill_score": "rating_skill_score",
    "weakness_score": "rating_weakness_score",
    "is_weak": "rating_is_weak"
}).copy()

# Keep only the columns required by the recommendation system
rating_stats = rating_stats[
    [
        "user",
        "rating_bucket",
        "rating_solve_rate",
        "rating_expected_rate",
        "rating_performance_ratio",
        "rating_avg_rating",
        "rating_mastery",
        "rating_skill_score",
        "rating_weakness_score",
        "rating_is_weak"
    ]
].copy()

print("Rating feature table:", rating_stats.shape)

# Display a few rows to verify the structure
print("\nRating feature examples:")
print(rating_stats.head())


# In[9]:


# ============================================================
# PERSON 3 - STEP 4 (CORRECTED)
# Fix user ID data types and build candidate dataset
# ============================================================

# ------------------------------------------------------------
# 1. Standardize user IDs
# ------------------------------------------------------------
# The CSV stores user IDs as strings, while our test_user
# was interpreted as a floating-point number.
#
# Convert ALL user IDs to strings so merges work correctly.

interactions["user"] = interactions["user"].astype(str)
user_stats["user"] = user_stats["user"].astype(str)
topic_aggregated["user"] = topic_aggregated["user"].astype(str)
rating_stats["user"] = rating_stats["user"].astype(str)

# Set the test user as a string
test_user = "-1e11"

print("Test User:", test_user)


# ------------------------------------------------------------
# 2. Find problems already attempted by the test user
# ------------------------------------------------------------

attempted_problems = set(
    interactions.loc[
        interactions["user"] == test_user,
        "problem_id"
    ]
)

print("Problems already attempted:", len(attempted_problems))


# ------------------------------------------------------------
# 3. Create candidate problems
# ------------------------------------------------------------
# Remove all problems that the user has already attempted.

candidate_data = problems[
    ~problems["problem_id"].isin(attempted_problems)
].copy()

# Add the user ID to every candidate problem
candidate_data["user"] = test_user

print("Candidate problems:", len(candidate_data))


# ------------------------------------------------------------
# 4. Add user-level features
# ------------------------------------------------------------

candidate_data = candidate_data.merge(
    user_stats,
    on="user",
    how="left"
)

print("After user features:", candidate_data.shape)


# ------------------------------------------------------------
# 5. Add problem-level features
# ------------------------------------------------------------

problem_feature_columns = [
    "problem_id",
    "problem_interactions",
    "problem_solve_rate",
    "problem_avg_attempts"
]

candidate_data = candidate_data.merge(
    problem_stats[problem_feature_columns],
    on="problem_id",
    how="left"
)

print("After problem features:", candidate_data.shape)


# ------------------------------------------------------------
# 6. Add topic-level features
# ------------------------------------------------------------

candidate_data = candidate_data.merge(
    topic_aggregated,
    on=["user", "problem_id"],
    how="left"
)

# Fill missing topic information with zero
topic_columns = [
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count"
]

candidate_data[topic_columns] = (
    candidate_data[topic_columns].fillna(0)
)

print("After topic features:", candidate_data.shape)


# ------------------------------------------------------------
# 7. Create rating buckets
# ------------------------------------------------------------

rating_bucket_values = np.sort(
    rating_stats["rating_bucket"].unique()
)

candidate_data["rating_bucket"] = (
    candidate_data["rating"]
    .apply(
        lambda x: rating_bucket_values[
            np.abs(rating_bucket_values - x).argmin()
        ]
    )
)


# ------------------------------------------------------------
# 8. Add rating-level features
# ------------------------------------------------------------

candidate_data = candidate_data.merge(
    rating_stats,
    on=["user", "rating_bucket"],
    how="left"
)

print("After rating features:", candidate_data.shape)


# ------------------------------------------------------------
# 9. Fill missing rating features
# ------------------------------------------------------------

rating_columns = [
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]

candidate_data[rating_columns] = (
    candidate_data[rating_columns].fillna(0)
)


# ------------------------------------------------------------
# 10. Final check
# ------------------------------------------------------------

print("\nFinal candidate dataset shape:", candidate_data.shape)

print(
    "Missing rating values:",
    candidate_data[rating_columns].isna().sum().sum()
)

print("\nFirst 5 candidate problems:")
print(
    candidate_data[
        [
            "problem_id",
            "rating",
            "tags",
            "topic_weakness_score",
            "rating_solve_rate"
        ]
    ].head()
)


# In[10]:


# ============================================================
# PERSON 3 - STEP 5
# Create the exact 23-feature matrix for prediction
# ============================================================

# These are the EXACT features used when Person 2 trained
# the machine-learning models.
#
# The order is important and must remain unchanged.

feature_columns = [
    "user_solved_count",
    "user_total_attempts",
    "user_solve_rate",
    "user_avg_attempts",
    "user_avg_problem_rating",
    "user_max_problem_rating",
    "rating",
    "problem_interactions",
    "problem_solve_rate",
    "problem_avg_attempts",
    "topic_weakness_score",
    "topic_solve_rate",
    "topic_skill_score",
    "topic_mastery",
    "weak_topic_count",
    "rating_solve_rate",
    "rating_expected_rate",
    "rating_performance_ratio",
    "rating_avg_rating",
    "rating_mastery",
    "rating_skill_score",
    "rating_weakness_score",
    "rating_is_weak"
]


# ------------------------------------------------------------
# Check that all 23 features exist
# ------------------------------------------------------------

missing_features = [
    column
    for column in feature_columns
    if column not in candidate_data.columns
]

print("Missing features:", missing_features)


# Stop if any feature is missing
assert len(missing_features) == 0, (
    f"Missing required features: {missing_features}"
)


# ------------------------------------------------------------
# Create the candidate feature matrix
# ------------------------------------------------------------

X_candidates = candidate_data[
    feature_columns
].copy()


# ------------------------------------------------------------
# Clean the feature matrix
# ------------------------------------------------------------
# Replace infinity values and missing values with zero
# so that the ML model can make predictions safely.

X_candidates = (
    X_candidates
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ------------------------------------------------------------
# Verify the final feature matrix
# ------------------------------------------------------------

print("\nCandidate feature matrix shape:", X_candidates.shape)
print("Number of features:", X_candidates.shape[1])
print(
    "Total missing values:",
    X_candidates.isna().sum().sum()
)

print("\nFeature columns:")
for i, column in enumerate(feature_columns, start=1):
    print(f"{i}. {column}")


# In[20]:


# Load Person 2's trained model directly from its file location

import joblib

model_path = r"C:\Users\saksh\gradient_boosting_model.pkl"

gb_model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(gb_model))


# In[24]:


# Check whether the original training data is still available

print("X_train available:", "X_train" in globals())
print("y_train available:", "y_train" in globals())
print("X_test available:", "X_test" in globals())
print("y_test available:", "y_test" in globals())


# In[35]:


# Load the trained Gradient Boosting model
import joblib

model_path = r"C:\Users\saksh\gradient_boosting_model.pkl"

gb_model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(gb_model))


# In[36]:


# Verify that the loaded model is trained

from sklearn.utils.validation import check_is_fitted

check_is_fitted(gb_model)

print("The model is FITTED and ready for predictions!")


# In[37]:


# Step 7: Generate prediction scores for all candidate problems

# Predict the probability that the user will solve each candidate
candidate_data["prediction_score"] = gb_model.predict_proba(
    X_candidates
)[:, 1]

print("Predictions generated successfully!")
print("Number of predictions:", len(candidate_data))
print("Minimum score:", candidate_data["prediction_score"].min())
print("Maximum score:", candidate_data["prediction_score"].max())
print("Average score:", candidate_data["prediction_score"].mean())


# In[38]:


# Step 8: Rank all candidate problems by prediction score

ranked_candidates = candidate_data.sort_values(
    by="prediction_score",
    ascending=False
).reset_index(drop=True)

# Add recommendation rank
ranked_candidates["rank"] = ranked_candidates.index + 1

print("Ranking completed!")
print("Total ranked problems:", len(ranked_candidates))

# Display the top 10 candidates
ranked_candidates[
    ["rank", "problem_id", "rating", "tags", "prediction_score"]
].head(10)


# In[39]:


# Step 9: Inspect the top 20 recommendations
# along with difficulty and topic-related features

top_20 = ranked_candidates[
    [
        "rank",
        "problem_id",
        "rating",
        "tags",
        "prediction_score",
        "topic_weakness_score",
        "topic_solve_rate",
        "topic_skill_score",
        "topic_mastery",
        "weak_topic_count",
        "rating_solve_rate",
        "rating_mastery",
        "rating_weakness_score"
    ]
].head(20)

print("Top 20 recommendations:")
display(top_20)


# In[40]:


# Step 10: Create a clean final recommendation table

final_recommendations = ranked_candidates[
    [
        "rank",
        "problem_id",
        "rating",
        "tags",
        "prediction_score",
        "topic_weakness_score",
        "weak_topic_count"
    ]
].head(10).copy()

# Rename columns for readability
final_recommendations.columns = [
    "Rank",
    "Problem",
    "Rating",
    "Tags",
    "Prediction Score",
    "Topic Weakness",
    "Weak Topic Count"
]

# Display final recommendations
display(final_recommendations)


# In[42]:


# Step 11: Model performance comparison graph

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

ax = model_results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# In[43]:


# Step 12: Prediction score analysis

score_analysis = ranked_candidates["prediction_score"].describe()

print("Prediction Score Analysis")
print("-------------------------")
print(f"Minimum Score : {score_analysis['min']:.4f}")
print(f"Maximum Score : {score_analysis['max']:.4f}")
print(f"Mean Score    : {score_analysis['mean']:.4f}")
print(f"Median Score  : {score_analysis['50%']:.4f}")
print(f"Std. Deviation: {score_analysis['std']:.4f}")

# Count candidates in different score ranges
score_ranges = pd.cut(
    ranked_candidates["prediction_score"],
    bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0],
    labels=[
        "0–0.50",
        "0.50–0.70",
        "0.70–0.80",
        "0.80–0.90",
        "0.90–1.00"
    ],
    include_lowest=True
)

score_distribution = score_ranges.value_counts().sort_index()

display(
    score_distribution.reset_index(
        name="Number of Problems"
    ).rename(
        columns={"prediction_score": "Prediction Score Range"}
    )
)


# In[44]:


# Step 13: Prediction score distribution

plt.figure(figsize=(10, 6))

plt.hist(
    ranked_candidates["prediction_score"],
    bins=20
)

plt.title("Distribution of Prediction Scores")
plt.xlabel("Prediction Score")
plt.ylabel("Number of Candidate Problems")
plt.tight_layout()
plt.show()


# In[45]:


# Step 14: Top-10 recommendation score visualization

top_10 = ranked_candidates.head(10).copy()

plt.figure(figsize=(10, 6))

plt.bar(
    top_10["problem_id"].astype(str),
    top_10["prediction_score"]
)

plt.title("Prediction Scores of Top-10 Recommended Problems")
plt.xlabel("Problem")
plt.ylabel("Prediction Score")
plt.ylim(0.99, 1.00)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[46]:


# Step 15: Gradient Boosting feature importance

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": gb_model.feature_importances_
})

# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
top_features = feature_importance.head(15)

display(top_features)

# Plot
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Top 15 Feature Importances - Gradient Boosting")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# In[47]:


# Step 26: Rating distribution of Top-10 recommendations

top_10 = ranked_candidates.head(10).copy()

plt.figure(figsize=(10, 6))

plt.bar(
    top_10["problem_id"].astype(str),
    top_10["rating"]
)

plt.title("Ratings of Top-10 Recommended Problems")
plt.xlabel("Problem")
plt.ylabel("Problem Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[48]:


# Step 27: Analyze tags in the Top-10 recommendations

from collections import Counter

top_10 = ranked_candidates.head(10).copy()

# Extract all tags from the Top-10 problems
all_tags = []

for tags in top_10["tags"].dropna():
    all_tags.extend([tag.strip() for tag in str(tags).split(",")])

tag_counts = Counter(all_tags)

tag_analysis = pd.DataFrame(
    tag_counts.items(),
    columns=["Tag", "Number of Recommendations"]
).sort_values(
    by="Number of Recommendations",
    ascending=False
)

display(tag_analysis)

# Plot
plt.figure(figsize=(10, 6))

plt.bar(
    tag_analysis["Tag"],
    tag_analysis["Number of Recommendations"]
)

plt.title("Topic Distribution of Top-10 Recommended Problems")
plt.xlabel("Topic / Tag")
plt.ylabel("Number of Recommended Problems")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[49]:


# Model Comparison Results

print("=" * 65)
print("MODEL COMPARISON RESULTS")
print("=" * 65)

print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 65)

print(f"{'Logistic Regression':<22} "
      f"{0.7285:>10.4f} "
      f"{0.9818:>10.4f} "
      f"{0.7285:>10.4f} "
      f"{0.8364:>10.4f}")

print(f"{'Random Forest':<22} "
      f"{0.8523:>10.4f} "
      f"{0.9813:>10.4f} "
      f"{0.8614:>10.4f} "
      f"{0.9175:>10.4f}")

print(f"{'Gradient Boosting':<22} "
      f"{0.9550:>10.4f} "
      f"{0.9570:>10.4f} "
      f"{0.9976:>10.4f} "
      f"{0.9769:>10.4f}")

print("=" * 65)


# In[ ]:




