# src/monitoring.py
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import tensorflow

# Model Monitoring and Drift Detection function
def model_monitoring(model, X_val, y_val):
    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("n_estimators", 100)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", model.score(X_val, y_val))

    # Drift detection (simple version)
    initial_accuracy = model.score(X_val, y_val)
    new_accuracy = accuracy_score(y_val, model.predict(X_val))  # Simulate new data
    if initial_accuracy - new_accuracy > 0.05:
        print("Model performance has drifted. Retraining is recommended.")
    else:
        print("No significant drift detected.")

    print("Model Monitoring completed!")
