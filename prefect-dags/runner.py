from prefect import flow, task, get_run_logger
from events_clean import clean_events
from raw_events_to_interactions import raw_events_to_interactions


@task
def events_cleaner():
    """
    Cleans the data
    """
    print("Events cleaning started")
    clean_df = clean_events()
    return clean_df

@task
def raw_events_to_interactions():
    """
    Cleans the data
    """
    print("Raw events to interactions")
    interactions, users, polls = raw_events_to_interactions()
    return interactions, users, polls

@task
def events_cleaner():
    """
    Cleans the data
    """
    print("Events cleaning started")
    clean_df = clean_events()
    return clean_df


@flow(log_prints=True)
def run(repo_owner: str = "PrefectHQ", repo_name: str = "prefect"):
    """
    Run this dag daily with the mentioned tasks
    mlops/events_clean.py cleans the events data and dumps it in
    data/prepared/events.pkl
    ● mlops/raw_events_to_interactions.py converts events to interaction and
    extracts user and polls data in separate files:
    ○ data/prepared/interactions.pkl: each row is a user-poll. Events have been
    collapsed to form one score.
    ○ data/prepared/users.pkl
    ○ data/prepared/polls.pkl
    ● mlops/prepare_for_model.py does some more cleaning for modeling purpose and
    dumps the following files:
    """
    print("Time to start event cleaning")
    cleaned_data = events_cleaner()
    print("Function 2 := Raw events to interactions")
    cleaned_data = raw_events_to_interactions(cleaned_data)
    # cleaned_data = events_cleaner()
    # cleaned_data = events_cleaner()
    


if __name__ == "__main__":
    run()