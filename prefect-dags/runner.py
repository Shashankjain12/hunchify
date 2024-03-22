from prefect import flow, task
from events_clean import clean_events

@task(retries=2)
def get_repo_info(repo_owner: str, repo_name: str):
    pass


@task
def events_cleaner():
    """
    Cleans the data
    """
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
    cleaned_data = events_cleaner()
    


if __name__ == "__main__":
    run()