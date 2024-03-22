from prefect import flow, task
from events_clean import 

@task(retries=2)
def get_repo_info(repo_owner: str, repo_name: str):
    """Get info about a repo - will retry twice after failing"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    api_response = httpx.get(url)
    api_response.raise_for_status()
    repo_info = api_response.json()
    return repo_info


@task
def events_cleaner(repo_info: dict):
    """
    Cleans the data
    """

    return contributors


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