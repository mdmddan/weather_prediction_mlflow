from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("weather_prediction")

# Lista todos los runs
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Elimina cada run uno por uno
for run in runs:
    client.delete_run(run.info.run_id)