# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: hunch_assignment
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup

# %%
import os

import numpy as np
import pandas as pd
from skimpy import skim
from collections import defaultdict
from pandarallel import pandarallel

from utils.evaluate import convert_df_to_dict, eval_add_show
from utils.utils import get_polls_data_from_interaction_data, get_users_data_from_interaction_data


# %%
from IPython.display import display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
pandarallel.initialize()

# %% [markdown]
# # Read data

# %%
data_path = os.path.join("data", "prepared")

# %%
file_name = "interactions_relevant.pkl"
file_path = os.path.join(data_path, file_name)
interactions: pd.DataFrame = None # type: ignore
if interactions is None:
    interactions = pd.read_pickle(
        file_path,
    )
assert interactions.shape[0] == interactions.groupby(['user_code', 'poll_code']).ngroups
display(interactions.head())    

# %%
file_name = "users_relevant.pkl"
file_path = os.path.join(data_path, file_name)
users: pd.DataFrame = None # type: ignore
if users is None:
    users = pd.read_pickle(
        file_path,
    )
assert users.shape[0] == users["user_code"].nunique()    
display(users.head())   

# %%
file_name = "polls_relevant.pkl"
file_path = os.path.join(data_path, file_name)
polls: pd.DataFrame = None # type: ignore
if polls is None:
    polls = pd.read_pickle(
        file_path,
    )
assert polls.shape[0] == polls["poll_code"].nunique()    
display(polls.head())   


# %% [markdown]
# # Train test split

# %%
print("User data split:")
display(
    pd.crosstab(
        users["has_just_one_poll"],
        users["has_no_useful_identity_data"],
        rownames=["is new user"],
        colnames=["no identity data"],
        margins=True,
        normalize=True,
    ).apply(lambda x: (x * 100).round(2))
)

# %%
users_has_columns = users.columns[users.columns.str.contains("has_")].tolist()

if not all([x in interactions.columns for x in users_has_columns]):
    rows_before = interactions.shape[0]    
    interactions = interactions.merge(
        users[
            ["user_code", *users_has_columns]
        ],
        on="user_code",
        how="left",
    )
    rows_after = interactions.shape[0]
    assert rows_before == rows_after

group_name_dict = {
    (True, True): ("new", "no_identity_data"),
    (True, False): ("new", "with_identity_data"),
    (False, True): ("existing", "no_identity_data"),
    (False, False): ("existing", "with_identity_data"),
}

grouped_data = (
    interactions
    .groupby(["has_just_one_poll", "has_no_useful_identity_data"])
)

grouped_data_dict = {
    group_name_dict[group_name]: group_data for group_name, group_data in grouped_data
}

# %%
train_data_dict = {}
test_data_dict = {}

for key, value in grouped_data_dict.items():
    # value = value.sort_values(["createdAt"]).copy()
    if key[0] == "new":
        test_indexes = (
            value[value["event"] != "Impression"]   # because we want only interactive polls in test
            .groupby(["poll_code"])    
            .filter(lambda x: len(x) > 1)   # if the poll has just this one user, then that poll should be present in train for it to be available for recommendation
            .groupby(["poll_code"])
            .apply(lambda x: x.sample(frac=0.1, random_state=123))
            .reset_index(level=[0], drop=True)
            .index
        )
    else:
        test_indexes = (
            value[value["event"] != "Impression"]   # because we want only interactive polls in test
            .groupby(["poll_code"])
            .filter(lambda x: len(x) > 2)  # to ensure that test polls are in train too
            .sample(frac=0.2, random_state=123)
            .index
        )
    test_data_dict[key] = value.loc[test_indexes]
    train_data_dict[key] = value.drop(test_indexes)

# %%
train_data = pd.concat(train_data_dict.values(), keys=train_data_dict.keys()).reset_index(drop=True)
train_users = get_users_data_from_interaction_data(train_data.copy())
train_polls = get_polls_data_from_interaction_data(train_data.copy())

assert set(train_data["user_code"].unique()).issubset(set(train_users["user_code"]))
assert set(train_users["user_code"]).issubset(train_data["user_code"].unique())
assert set(train_data["poll_code"].unique()).issubset(set(train_polls["poll_code"]))
assert set(train_polls["poll_code"]).issubset(train_data["poll_code"].unique())

print("train:")
print(f"{train_data.shape}")
print(f"""Users in train data: {len(train_data["user_code"].unique())}""")
print(f"""Polls in train data: {len(train_data["poll_code"].unique())}""")
print("User-poll interaction split:")
display(
    pd.crosstab(
        train_data["has_just_one_poll"],
        train_data["has_no_useful_identity_data"],
        rownames=["is new user"],
        colnames=["no identity data"],
        margins=True,
        normalize=True,
    ).apply(lambda x: (x * 100).round(2))
)

test_data = pd.concat(test_data_dict.values(), keys=test_data_dict.keys()).reset_index(drop=True)
test_users = get_users_data_from_interaction_data(test_data.copy())
test_polls = get_polls_data_from_interaction_data(test_data.copy())

assert set(test_data["user_code"].unique()).issubset(set(test_users["user_code"]))
assert set(test_users["user_code"]).issubset(test_data["user_code"].unique())
assert set(test_data["poll_code"].unique()).issubset(set(test_polls["poll_code"]))
assert set(test_polls["poll_code"]).issubset(test_data["poll_code"].unique())

print("\ntest:")
print(f"{test_data.shape}")
print(f"""Users in test data: {len(test_data["user_code"].unique())}""")
print(f"""Polls in test data: {len(test_data["poll_code"].unique())}""")
print("User-poll interaction split:")
display(
    pd.crosstab(
        test_data["has_just_one_poll"],
        test_data["has_no_useful_identity_data"],
        rownames=["is new user"],
        colnames=["no identity data"],
        margins=True,
        normalize=True,
    ).apply(lambda x: (x * 100).round(2))
)

print(
    f"""\nTest users in train: {np.isin(test_data["user_code"].unique(), train_data["user_code"].unique()).sum() / len(test_data["user_code"].unique())* 100:.2f}%"""
)
print(
    f"""Test polls in train: {np.isin(test_data["poll_code"].unique(), train_data["poll_code"].unique()).sum() / len(test_data["poll_code"].unique())* 100:.2f}%"""
)


assert train_data.shape[0] + test_data.shape[0] == interactions.shape[0]

# %%
print("Distribution of polls per user in train")
display(train_users["n_interactive_polls"].describe(np.arange(0.1, 1, 0.1)).to_frame().T)

print("Distribution of polls per user in test")
display(test_users["n_interactive_polls"].describe(np.arange(0.1, 1, 0.1)).to_frame().T)

# %%
assert interactions.shape[0] == interactions.groupby(["poll_code", "user_code"]).ngroups
assert users.shape[0] == users["user_code"].nunique()
assert polls.shape[0] == polls["poll_code"].nunique()

assert set(interactions["user_code"].unique()).issubset(set(users["user_code"]))
assert set(users["user_code"]).issubset(interactions["user_code"].unique())
all_users = users["user_code"]

assert set(interactions["poll_code"].unique()).issubset(set(polls["poll_code"]))
assert set(polls["poll_code"]).issubset(interactions["poll_code"].unique())
all_polls = polls["poll_code"]

# %%
assert train_data.shape[0] == train_data.groupby(["poll_code", "user_code"]).ngroups
assert train_users.shape[0] == train_users["user_code"].nunique()
assert train_polls.shape[0] == train_polls["poll_code"].nunique()

assert set(train_data["user_code"].unique()).issubset(set(train_users["user_code"]))
assert set(train_users["user_code"]).issubset(train_data["user_code"].unique())
train_users_users = train_users["user_code"]


assert set(train_data["poll_code"].unique()).issubset(set(train_polls["poll_code"]))
assert set(train_polls["poll_code"]).issubset(train_data["poll_code"].unique())
train_polls_polls = train_polls["poll_code"]

# %%
assert test_data.shape[0] == test_data.groupby(["poll_code", "user_code"]).ngroups
assert test_users.shape[0] == test_users["user_code"].nunique()
assert test_polls.shape[0] == test_polls["poll_code"].nunique()

assert set(test_data["user_code"].unique()).issubset(set(test_users["user_code"]))
assert set(test_users["user_code"]).issubset(test_data["user_code"].unique())
test_users_users = test_users["user_code"]


assert set(test_data["poll_code"].unique()).issubset(set(test_polls["poll_code"]))
assert set(test_polls["poll_code"]).issubset(test_data["poll_code"].unique())
test_polls_polls = test_polls["poll_code"]

# %% [markdown]
# # Data prep. for model

# %%
print(f"Total train users: {len(train_users)}")
print(f"Total train polls: {len(train_polls)}")

# get just those train polls that the user interacted with
train_data_i = train_data[train_data["event_score"] != 0].copy()
train_users_with_interactions = train_data_i["user_code"].unique()
train_polls_with_interactions = train_data_i["poll_code"].unique()

print(f"train users with interactions: {len(train_users_with_interactions)}")
print(f"train polls with interactions: {len(train_polls_with_interactions)}")

# %%
print(f"Total test users: {len(test_users)}")
print(f"Total test polls: {len(test_polls)}")

# convert it to dict format
test_data_dict = convert_df_to_dict(
    test_data[["user_code", "poll_code", "event_score"]].copy(), with_pred_rating=True
)

# %%
train_poll_codes_by_user = train_data.groupby("user_code")["poll_code"].agg(list).reset_index()
train_poll_codes_by_user.rename(columns={"poll_code": "train_poll_codes_list"}, inplace=True)

rows_before = test_data.shape[0]

test_data = test_data.merge(train_poll_codes_by_user, on="user_code", how="left")
test_data["train_poll_codes_list"] = test_data["train_poll_codes_list"].apply(
    lambda d: d if isinstance(d, list) else []
)
rows_after = test_data.shape[0]

assert rows_before == rows_after

# %%
test_users_in_train = test_users[np.isin(test_users, train_users)].copy()

test_data_in_train = (
    test_data[test_data["user_code"].isin(test_users_in_train)].copy().reset_index(drop=True)
)

test_data_in_train_dict = (
    test_data_in_train.groupby("user_code")[["poll_code", "event_score"]]
    .apply(lambda g: list(map(tuple, g.values)))
    .to_dict()
)

for user_code, recommendation in test_data_in_train_dict.items():
    recommendation = dict(sorted(recommendation.items(), key=lambda x: x[1], reverse=True))
    test_data_in_train_dict[user_code] = recommendation

# %% [markdown]
# # Modeling

# %%
model_results_comparison = pd.DataFrame()

# %% [markdown]
# ## Basline-0: same top popular polls to every user

# %%
test_polls_set = set(test_polls)

for popularity_metric in [
    "n_interactive_users",
    # "event_score_sum_by_poll",
    # "event_score_by_poll_per_interactive_user",
]:
    recommended_polls = (
        train_polls.sort_values(popularity_metric, ascending=False)["poll_code"].reset_index(
            drop=True
        )
        .copy()
        .to_list()
    )

    df_recommended = test_users[["user_code"]].copy()

    rows_before = df_recommended.shape[0]
    df_recommended = df_recommended.merge(
        test_data[["user_code", "train_poll_codes_list"]].drop_duplicates(["user_code"]),
        on="user_code",
        how="left",
    )
    rows_after = df_recommended.shape[0]
    assert rows_before == rows_after

    for n in [10, 25, 50]:
        model_name = f"Baseline: Top {n} polls by " + popularity_metric
        df_recommended["recommended_polls"] = [recommended_polls] * df_recommended.shape[0]
        df_recommended["recommended_polls_filtered"] = df_recommended.parallel_apply(
            lambda x: x["recommended_polls"][0:n],
            axis=1,
        ) # type: ignore

        # df_recommended["recommended_polls_filtered"] = df_recommended.parallel_apply(
        #     lambda x: [
        #         poll for poll in x["recommended_polls"] if poll not in x["train_poll_codes_list"]
        #     ][0:n],
        #     axis=1,
        # ) # type: ignore

        recommendation_dict = df_recommended.set_index("user_code")[
            "recommended_polls_filtered"
        ].to_dict()

        (
            ndcg_by_user,
            precision_by_user,
            recall_by_user,
            results,
            model_results_comparison,
        ) = eval_add_show(
            model_name,
            recommendation_dict,
            test_data_dict,
            all_polls,
            train_data[["user_code", "poll_code", "event_score"]].copy(),
            with_pred_rating=False,
            model_results_comparison=model_results_comparison.copy(),
            add=True,
            show=False,
        )
with pd.option_context("display.float_format", "{:,.2%}".format):
    display(model_results_comparison)

# %% [markdown]
# ## SVD from `surprise` package

# %%
from surprise import Dataset, Reader, SVD

# %%
rating_min = train_data["event_score"].min()
rating_max = train_data["event_score"].max()
print(f"Min rating: {rating_min}, Max rating: {rating_max}")
req_cols = ["user_code", "poll_code", "event_score"]

reader = Reader(rating_scale=(rating_min, rating_max))
train_data_surprise = Dataset.load_from_df(train_data[req_cols], reader).build_full_trainset()

# %%
algo = SVD(n_factors=10, n_epochs=40, lr_all=0.005, reg_all=0.1)
algo.fit(train_data_surprise)


# %%
def get_test_predictions(algo, test_data, all_polls, is_filter_train_polls_out=False, n=50):
    test_predictions_by_user = defaultdict(list)
    for user_code in test_data["user_code"].unique():
        train_poll_codes_list = test_data.loc[
            test_data["user_code"] == user_code, "train_poll_codes_list"
        ].values.tolist()[0]
        if is_filter_train_polls_out:
            candidate_polls = [poll for poll in all_polls if poll not in train_poll_codes_list]
        else:
            candidate_polls = all_polls
        for poll_code in candidate_polls:
            predicted_rating = algo.predict(user_code, poll_code).est
            test_predictions_by_user[user_code].append((poll_code, predicted_rating))

    for user_code, recommendation in test_predictions_by_user.items():
        recommendation.sort(key=lambda x: x[1], reverse=True)
        test_predictions_by_user[user_code] = recommendation[:n]

    return test_predictions_by_user


# %%
for is_filter_train_polls_out in [False, True]:
    for n in [10, 25, 50]:
        test_predictions_by_user_topn = get_test_predictions(
            algo, test_data, all_polls, is_filter_train_polls_out=is_filter_train_polls_out, n=n
        )
        model_name = f"SVD: {n} - has_train_polls {is_filter_train_polls_out}"
        ndcg_by_user, precision_by_user, recall_by_user, results, model_results_comparison = eval_add_show(
            model_name,
            test_predictions_by_user_topn,
            test_data_dict,
            all_polls,
            train_data[["user_code", "poll_code", "event_score"]].copy(),
            with_pred_rating=True,
            model_results_comparison=model_results_comparison.copy(),
            add=True,
            show=False,
        )

# %%
with pd.option_context("display.float_format", "{:,.2%}".format):
    display(model_results_comparison)

# %%
