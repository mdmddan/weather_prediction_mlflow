# load_best_model.py
import mlflow
import mlflow.sklearn
import pandas as pd

# Define el nombre del experimento que usaste
experiment_name = "weather_prediction"

# Obtiene el experimento por nombre
experiment = mlflow.get_experiment_by_name(experiment_name)

# Busca todos los runs de ese experimento, ordenados por el menor rmse_test
experiment_results = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse_test ASC"],  # Ordena del mejor (menor RMSE) al peor
    max_results=1                        # Solo trae el mejor run
)



run_id = experiment_results.iloc[0]["run_id"]
model_uri = f"runs:/{run_id}/model"

# Cargar el modelo
best_model = mlflow.sklearn.load_model(model_uri)
print("✅ Modelo cargado correctamente.")

# Ejemplo de predicción:
X_new = pd.DataFrame({
    "cloud_cover": [4],
    "sunshine": [5],
    "global_radiation": [100],
    "precipitation": [0],
    "pressure": [101300],
    "snow_depth": [0]
})

y_pred = best_model.predict(X_new)
print(f"Predicción de temperatura: {y_pred[0]:.2f} °C")
