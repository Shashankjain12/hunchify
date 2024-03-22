from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/Shashankjain12/hunchify/tree/main/prefect-dags",
        entrypoint="runner.py:run",
    ).deploy(
        name="hunch-deployment",
        work_pool_name="my-managed-pool",
        cron="0 1 * * *",
    )