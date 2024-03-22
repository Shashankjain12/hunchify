import numpy as np
import pandas as pd

def get_users_data_from_interaction_data(events: pd.DataFrame):
    users = (
        events.groupby(["user_code"])
        .agg(
            country=("country", "first"),
            city_code=("city_code", "first"),
            gender=("gender", "first"),
            age=("age", "first"),
            college_code=("college_code", "first"),
            n_polls=("poll_code", "nunique"),
            event_score_sum_by_user=("event_score", "sum"),
        )
        .reset_index()
    )
    users_polls_by_event = (
        events.groupby(["user_code"])
        .apply(
            lambda group: pd.Series(
                {
                    "n_interactive_polls": group.loc[
                        group["event"] != "Impression", "poll_code"
                    ].nunique(),
                    "n_answered_polls": group.loc[
                        group["event"] == "Polls Answered", "poll_code"
                    ].nunique(),
                    "n_expands_polls": group.loc[group["event"] == "Expand", "poll_code"].nunique(),
                    "n_shares_polls": group.loc[group["event"] == "Shares", "poll_code"].nunique(),
                }
            )
        )
        .reset_index()
    )
    users = users.merge(users_polls_by_event, on="user_code", how="left")

    users["n_polls_coverage"] = users["n_polls"] / events["poll_code"].nunique()
    users["n_interactive_polls_coverage"] = (
        users["n_interactive_polls"] / events["poll_code"].nunique()
    )
    users["n_interactive_polls_proportion"] = users["n_interactive_polls"] / users["n_polls"]
    users["event_score_by_user_per_interactive_poll"] = np.where(
        users["n_interactive_polls"] == 0,
        0,
        users["event_score_sum_by_user"] / users["n_interactive_polls"],
    )
    users["has_just_one_poll"] = users["n_polls"] == 1
    users["has_no_interactive_polls"] = users["n_interactive_polls"] == 0

    users["has_no_useful_location_data"] = (
        (users["country"].isna()) | (users["country"] == "country_1")
    ) & (users["city_code"].isna())

    users["has_no_useful_identity_data"] = (
        (users["gender"].isna()) & (users["age"].isna()) & (users["college_code"].isna())
    )

    users["has_no_useful_user_data"] = (
        users["has_no_useful_location_data"] & users["has_no_useful_identity_data"]
    )

    return users

def get_polls_data_from_interaction_data(events):
    polls = (
        events.groupby(["poll_code"])
        .agg(
            n_users=("user_code", "nunique"),
            event_score_sum_by_poll=("event_score", "sum"),
        )
        .reset_index()
    )
    polls_users_by_event = events.groupby(["poll_code"]).apply(
        lambda group: pd.Series(
            {
                "n_interactive_users": group.loc[
                    group["event"] != "Impression", "user_code"
                ].nunique(),
                "n_answered_users": group.loc[
                    group["event"] == "Polls Answered", "user_code"
                ].nunique(),
                "n_expands_users": group.loc[group["event"] == "Expand", "user_code"].nunique(),
                "n_shares_users": group.loc[group["event"] == "Shares", "user_code"].nunique(),
            }
        )
    )
    polls = polls.merge(polls_users_by_event, on="poll_code", how="left")

    polls["n_users_coverage"] = polls["n_users"] / events["user_code"].nunique()
    polls["n_interactive_users_coverage"] = (
        polls["n_interactive_users"] / events["user_code"].nunique()
    )
    polls["n_interactive_users_proportion"] = polls["n_interactive_users"] / polls["n_users"]
    polls["event_score_by_poll_per_interactive_user"] = np.where(
        polls["n_interactive_users"] == 0,
        0,
        polls["event_score_sum_by_poll"] / polls["n_interactive_users"],
    )
    polls["has_no_interactive_users"] = polls["n_interactive_users"] == 0

    return polls



