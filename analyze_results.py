# analyze_results.py

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# 1. Buscar los experimentos
experiment = mlflow.get_experiment_by_name("weather_prediction")

experiment_results = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse_test ASC"]
)

print("\nTop experiment results:")
print(experiment_results[["run_id", "params.model", "metrics.rmse_train", "metrics.rmse_test", "metrics.rmse_gap"]])

# 2. Guardar los resultados a CSV
experiment_results.to_csv("experiment_results.csv", index=False)
print("✅ Resultados exportados a experiment_results.csv")

# 3. Graficar los resultados
plt.figure(figsize=(8, 5))
plt.scatter(experiment_results['params.model'], experiment_results['metrics.rmse_test'])
plt.xlabel('Modelo')
plt.ylabel('RMSE Test')
plt.title('Comparación de modelos por RMSE en Test')
plt.show()
plt.savefig("experiment_results_plot.png")
print("✅ Gráfica guardada como experiment_results_plot.png")

# Filtrar las columnas que queremos mostrar en la tabla
table_results = experiment_results[["params.model", "metrics.rmse_train", "metrics.rmse_test", "metrics.rmse_gap"]]

# Renombrar columnas para que se vea bien en la tabla
table_results = table_results.rename(columns={
    "params.model": "Model",
    "metrics.rmse_train": "RMSE Train",
    "metrics.rmse_test": "RMSE Test",
    "metrics.rmse_gap": "RMSE Gap"
})

# Ordenar por RMSE Test
table_results = table_results.sort_values(by="RMSE Test")

# Exportar la tabla en formato Markdown para el README
with open("results_table.md", "w") as f:
    f.write(table_results.to_markdown(index=False))

print("✅ Markdown table exported to results_table.md")
