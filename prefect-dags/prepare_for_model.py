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

import pandas as pd
from skimpy import skim


# %%
from IPython.display import display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %% [markdown]
# # Read data

# %%
data_path = os.path.join("data", "prepared")

# %%
file_name = "interactions.pkl"
file_path = os.path.join(data_path, file_name)
interactions: pd.DataFrame = None # type: ignore
if interactions is None:
    interactions = pd.read_pickle(
        file_path,
    )
assert interactions.shape[0] == interactions.groupby(['user_code', 'poll_code']).ngroups
display(interactions.head())    

# %%
file_name = "users.pkl"
file_path = os.path.join(data_path, file_name)
users: pd.DataFrame = None # type: ignore
if users is None:
    users = pd.read_pickle(
        file_path,
    )
assert users.shape[0] == users["user_code"].nunique()    
display(users.head())   

# %%
file_name = "polls.pkl"
file_path = os.path.join(data_path, file_name)
polls: pd.DataFrame = None # type: ignore
if polls is None:
    polls = pd.read_pickle(
        file_path,
    )
assert polls.shape[0] == polls["poll_code"].nunique()    
display(polls.head())   


# %% [markdown]
# # Delete irrelevant users and polls

# %% [markdown]
# Users with 0% interactive polls are not relevant for our purpose because we can neither learn anything from them, nor use them in the test set. Same goes for polls with 0% interactive users. So, let's delete them.

# %%
users_with_zero_interactive_polls = users.loc[
    users["has_no_interactive_polls"], "user_code"
].values
print(
    f"Users with zero interactive polls: {len(users_with_zero_interactive_polls)} ({len(users_with_zero_interactive_polls)/users.shape[0]*100:.2f}%)"
)

polls_with_zero_interactive_users = polls.loc[
    polls["has_no_interactive_users"], "poll_code"
].values
print(
    f"Polls with zero interactive users: {len(polls_with_zero_interactive_users)} ({len(polls_with_zero_interactive_users)/polls.shape[0]*100:.2f}%)"
)

# %%
users = users[~users["has_no_interactive_polls"]].copy().reset_index(drop=True)
users.drop(columns=["has_no_interactive_polls"], inplace=True)

assert users.shape[0] == users["user_code"].nunique()

polls = polls[~polls["has_no_interactive_users"]].copy().reset_index(drop=True)
polls.drop(columns=["has_no_interactive_users"], inplace=True)

assert polls.shape[0] == polls["poll_code"].nunique()

n_polls = polls.shape[0]
n_users = users.shape[0]

print(f"n_users: {n_users}")
print(f"n_polls: {n_polls}")

# %%
is_non_interactive_user = interactions["user_code"].isin(users_with_zero_interactive_polls)
is_non_interactive_poll = interactions["poll_code"].isin(polls_with_zero_interactive_users)

is_non_interactive = is_non_interactive_user | is_non_interactive_poll

print(
    f"User-poll interactions to be removed: {is_non_interactive.sum()} ({is_non_interactive.sum()/interactions.shape[0]*100:.2f}%)"
)

# %%
rows_before = interactions.shape[0]
interactions = interactions[~is_non_interactive].copy().reset_index(drop=True)
rows_after = interactions.shape[0]
assert rows_before - rows_after == is_non_interactive.sum()

# %%
assert set(set(interactions["user_code"].unique())).issubset(set(users["user_code"]))
assert set(set(users["user_code"])).issubset(set(interactions["user_code"].unique()))

assert set(set(interactions["poll_code"].unique())).issubset(set(polls["poll_code"]))
assert set(set(polls["poll_code"])).issubset(set(interactions["poll_code"].unique()))

# %% [markdown]
# ## Concise Summary:

# %%
print(f"Total number of unique user-poll interactions: {interactions.shape[0]}")
print(
    f"""\nDistribution by event type:\n{interactions["event"].value_counts(dropna=False, normalize=True)}"""
)
n_polls = interactions["poll_code"].nunique()
n_users = interactions["user_code"].nunique()
print(f"\nn_users: {n_users}")
print(f"n_polls: {n_polls}")
skim(
    interactions.apply(lambda x: x.astype("category") if x.dtype == "object" else x)
)

# %% [markdown]
# # Write

# %%
pd.to_pickle(interactions, os.path.join(data_path, "interactions_relevant.pkl"))
pd.to_pickle(users, os.path.join(data_path, "users_relevant.pkl"))
pd.to_pickle(polls, os.path.join(data_path, "polls_relevant.pkl"))
