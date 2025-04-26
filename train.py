# train.py

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import mlflow
import mlflow.sklearn
from math import sqrt
from datetime import datetime

# -----------------------
# Preprocesamiento
# -----------------------
def preprocess_data(filepath):
    df = pd.read_csv(filepath).dropna()
    features = ['cloud_cover', 'sunshine', 'global_radiation',
                'precipitation', 'pressure', 'snow_depth']
    X = df[features]
    y = df['mean_temp']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Naming del run
# -----------------------
def build_run_name(model_name, param_dist, best_params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not param_dist:
        return f"{model_name}__NoSearch__{timestamp}"
    params_str = "_".join([f"{k}={v}" for k, v in best_params.items()])
    return f"{model_name}__RandomSearch__{params_str}__{timestamp}"

# -----------------------
# Entrenamiento + tracking con métricas Train y Test
# -----------------------
def train_model(model, param_dist, X_train, X_test, y_train, y_test, model_name, experiment_name="weather_prediction"):
    mlflow.set_experiment(experiment_name)

    if param_dist:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring=make_scorer(mean_squared_error, squared=False),  # RMSE
            n_iter=10,
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        rmse_train = search.best_score_  # Esto es el RMSE del CV (promedio de los folds)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
        best_model = model
        best_params = {}

    # Calculamos el RMSE sobre el test set
    y_test_pred = best_model.predict(X_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))

    # Nombre del run
    run_name = build_run_name(model_name, param_dist, best_params)

    # Tracking con MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("rmse_gap", rmse_test - rmse_train)  # Para monitorear overfitting

        # NUEVO: input_example y signature
        from mlflow.models.signature import infer_signature
        input_example = X_train.iloc[:1]
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            pip_requirements="requirements.txt"
        )

        print(f"{run_name} - RMSE Train: {rmse_train} | RMSE Test: {rmse_test}")

    return best_model, rmse_train, rmse_test

# -----------------------
# Modelos y parámetros
# -----------------------
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "param_grid": {}  # No hay hiperparámetros importantes en LinearRegression básica
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "max_depth": list(range(3, 21)),                       # Profundidades controladas para evitar overfitting
            "min_samples_split": [2, 5, 10, 20],                  # Ajuste fino de splits
            "min_samples_leaf": [1, 2, 5, 10],                    # Control de tamaño mínimo de hojas
            "max_features": ["sqrt", "log2", None]                # Buenas prácticas para árboles (Breiman)
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 500],                      # Balance entre estabilidad y tiempo de cómputo
            "max_depth": [None, 5, 10, 20],                       # Profundidad controlada
            "min_samples_split": [2, 5, 10, 20],                  # Para evitar overfitting en datasets medianos a grandes
            "min_samples_leaf": [1, 2, 5, 10],                    # Igual que en DecisionTree
            "max_features": ["sqrt", "log2", None],               # Uso estándar para RandomForest
            "bootstrap": [True, False]                            # Bagging activado/desactivado para evaluar el impacto
        }
    }
}

# -----------------------
# Pipeline principal
# -----------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("data/london_weather.csv")

    for model_name, mp in models.items():
        print(f"\nEntrenando {model_name}...")
        train_model(
            model=mp["model"],
            param_dist=mp["param_grid"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name
        )

    # Buscar los experimentos
    experiment = mlflow.get_experiment_by_name("weather_prediction")
    experiment_results = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_test ASC"]
    )

    print("\nTop experiment results:")
    print(experiment_results[["run_id", "params.model", "metrics.rmse_train", "metrics.rmse_test", "metrics.rmse_gap"]])

