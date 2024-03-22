from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/Shashankjain12/hunchify.git",
        entrypoint="prefect-dags/runner.py:run",
    ).deploy(
        name="hunch-deployment",
        work_pool_name="my-managed-pool",
        cron="5 * * * *",
    )