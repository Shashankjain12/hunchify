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
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimpy import skim
# %%
from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# %% [markdown]
# # Read raw data

def clean_events():
# %%
    data_path = os.path.join("data", "raw")
    file_name = "events.pkl"
    file_path = os.path.join(data_path, file_name)

    # %%
    events_raw: pd.DataFrame = None  # type: ignore
    if events_raw is None:
        events_raw = pd.read_pickle(file_path)

    # %%
    display(events_raw.head())
    display(events_raw.info())

    # %%
    skim(events_raw.apply(lambda x: x.astype("category") if x.dtype == "object" else x))

    # %% [markdown]
    # # Make a backup

    # %%
    events = events_raw.copy()

    # %% [markdown]
    # # Clean

    # %% [markdown]
    # ## Handling potential duplicates

    # %%
    temp = (
        events.groupby(["user_code", "poll_code", "event", "createdAt"])
        .filter(lambda x: x.shape[0] > 1)
        .sort_values(by=["event", "user_code", "poll_code", "createdAt"])
    )

    # %%
    print(f"No. of potential duplicates: {temp.shape[0]}")
    for col in ["user_code", "poll_code", "event"]:
        print(f"\n{col}: {temp[col].value_counts(dropna=False, normalize=True).iloc[:5]}")
    display(temp.groupby(["event"]).head())

    # %% [markdown]
    # Different options for some reason! Does not make sense. Let's drop them.

    # %%
    primary_key = ["user_code", "poll_code", "event", "createdAt"]

    events.drop_duplicates(subset=primary_key, ignore_index=True, inplace=True)

    assert events.shape[0] == events.groupby(primary_key).ngroups

    # %% [markdown]
    # ## Handle rows with same user, poll, and event

    # %%
    events.sort_values(["createdAt"], inplace=True)

    is_same_event = events.duplicated(subset=["user_code", "poll_code", "event"], keep="last")

    print(
        f"No. of interactions to be removed: {is_same_event.sum()} ({is_same_event.sum()/events.shape[0]*100:.2f}%)"
    )

    # %%
    events = events[~is_same_event].copy().reset_index(drop=True)

    # %% [markdown]
    # # Write

    # %%
    print(events.shape)
    display(events.head())

    # %%
    data_path = os.path.join("data", "prepared")
    file_name = "events.pkl"
    file_path = os.path.join(data_path, file_name)

    # %%
    pd.to_pickle(events, file_path)