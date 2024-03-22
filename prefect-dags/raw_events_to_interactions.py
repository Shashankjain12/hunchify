import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimpy import skim
import plotly.express as px
import seaborn as sns
from box import Box
from fuzzywuzzy import fuzz, process
from collections import defaultdict
import pickle
from itertools import product
from pandarallel import pandarallel

from utils.utils import get_polls_data_from_interaction_data, get_users_data_from_interaction_data


from IPython.display import display


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


pandarallel.initialize()

def raw_events_to_interactions():

    data_path = os.path.join("data", "prepared")
    file_name = "events.pkl"
    file_path = os.path.join(data_path, file_name)

    events: pd.DataFrame = None  
    if events is None:
        events = pd.read_pickle(
            file_path,
        )


    display(events.head())

    skim(events.apply(lambda x: x.astype("category") if x.dtype == "object" else x))

    event_score_dict = {"Impression": 0, "Expand": 1, "Polls Answered": 2, "Shares": 3}
    events["event_score"] = events["event"].map(event_score_dict)

    users = get_users_data_from_interaction_data(events.copy())

    display(users.head())


    temp = users["n_interactive_polls"].value_counts(sort=False).sort_index()
    print(
        f"""Users with no interactions, just impressions: {temp[0]} ({(temp[0] / users.shape[0] * 100):.2f}%)"""
    )

    temp = users["n_polls"].value_counts(sort=False).sort_index()
    print(f"""Users with just 1 poll: {temp[1]} ({(temp[1] / users.shape[0] * 100):.2f}%)""")


    print(
        f"""Users with no useful location data: {users["has_no_useful_location_data"] .sum()} ({(users["has_no_useful_location_data"] .sum() / users.shape[0] * 100):.2f}%)"""
    )


    print(
        f"""Users with no useful identity data: {users["has_no_useful_identity_data"].sum()} ({(users["has_no_useful_identity_data"].sum() / users.shape[0] * 100):.2f}%)"""
    )

    print(
        f"""Users with no useful user data: {users["has_no_useful_user_data"].sum()} ({(users["has_no_useful_user_data"].sum() / users.shape[0] * 100):.2f}%)"""
    )

    skim(users.apply(lambda x: x.astype("category") if x.dtype == "object" else x))

    for col in ["country", "city_code", "gender", "college_code"]:
        counts = users[col].value_counts(dropna=True, normalize=True).reset_index()
        counts["proportion_cumulative"] = counts["proportion"].cumsum().div(counts["proportion"].sum())
        index = (
            counts.loc[counts["proportion_cumulative"] > 0.9, "proportion_cumulative"].idxmin()
        ) + 1
        index = max(index, 5)
        print(f"\n{counts.iloc[0:index, 0:2]}")

    for col in [
        "age",
        "n_polls",
        "n_interactive_polls_proportion",
        "event_score_by_user_per_interactive_poll",
    ]:
        display(
            users[col].describe(
                percentiles=np.concatenate(
                    [
                        np.arange(0.01, 0.06, 0.01),
                        [0.1],
                        np.arange(0.25, 0.8, 0.25),
                        [0.9],
                        np.arange(0.95, 0.99, 0.01),
                    ]
                )
            )
        )

    is_younger_than_teen = users["age"] < 13
    is_older_than_40 = users["age"] > 40

    is_invalid_age = is_younger_than_teen | is_older_than_40

    users["age"] = users["age"].where(~is_invalid_age)

    users["age"].describe(
        percentiles=np.concatenate(
            [
                np.arange(0.01, 0.06, 0.01),
                [0.1],
                np.arange(0.25, 0.8, 0.25),
                [0.9],
                np.arange(0.95, 0.99, 0.01),
            ]
        )
    )

    bins = [0, 16, 18, 22, 25, 30, 40]
    labels = [f"({bins[i]}-{bins[i+1]}]" for i in range(len(bins) - 1)]
    print(labels)
    users["age_binned"] = pd.cut(
        users["age"], bins=bins, labels=labels, right=True, include_lowest=False
    ).astype("object")


    users["age_binned"].value_counts(dropna=True, normalize=True)

    min_code_proportion = 0.01
    for col in ["city_code", "college_code"]:
        code_propoptions = users[col].value_counts(dropna=True, normalize=True)
        codes_to_replace = code_propoptions[code_propoptions < min_code_proportion].index
        users[col + "_trimmed"] = users[col].replace(codes_to_replace, "Other").copy()
        print(users[col + "_trimmed"].value_counts(dropna=True, normalize=True))

    users["country_filled"] = users["country"].copy()
    users["gender_filled"] = users["gender"].copy()

    for col in [
        "country_filled",
        "gender_filled",
        "age_binned",
        "city_code_trimmed",
        "college_code_trimmed",
    ]:
        users[col].fillna("Missing", inplace=True)

    polls = get_polls_data_from_interaction_data(events.copy())

    display(polls.head())

    skim(polls.apply(lambda x: x.astype("category") if x.dtype == "object" else x))

    for col in [
        "n_users",
        "n_interactive_users_proportion",
        "event_score_by_poll_per_interactive_user",
    ]:
        display(
            polls[col].describe(
                percentiles=np.concatenate(
                    [
                        np.arange(0.01, 0.06, 0.01),
                        [0.1],
                        np.arange(0.25, 0.8, 0.25),
                        [0.9],
                        np.arange(0.95, 0.99, 0.01),
                    ]
                )
            )
        )

    events.sort_values(["event_score"], inplace=True)

    is_same_user_poll = events.duplicated(subset=["user_code", "poll_code"], keep="last")

    print(
        f"No. of different events for same user-poll to be collapsed: {is_same_user_poll.sum()} ({is_same_user_poll.sum()/events.shape[0]*100:.2f}%)"
    )

    interactions = events[~is_same_user_poll].copy().reset_index(drop=True)
    assert interactions.shape[0] == interactions.groupby(["poll_code", "user_code"]).ngroups
    primary_key = ["user_code", "poll_code"]
    skim(interactions.apply(lambda x: x.astype("category") if x.dtype == "object" else x))

    pd.to_pickle(interactions, os.path.join(data_path, "interactions.pkl"))
    pd.to_pickle(users, os.path.join(data_path, "users.pkl"))
    pd.to_pickle(polls, os.path.join(data_path, "polls.pkl"))
    return interactions, users, polls