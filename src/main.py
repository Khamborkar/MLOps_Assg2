# main.py
import pandas as pd
import tensorflow
from tensorflow.keras.datasets import fashion_mnist
from src.eda import generate_eda_report
from src.feature_engineering import feature_engineering
from src.automl import automl_optimization
from src.monitoring import model_monitoring

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert dataset to Pandas DataFrame (subset for efficiency)
subset_size = 500  
train_df = pd.DataFrame(train_images[:subset_size].reshape(subset_size, -1))
train_df['label'] = train_labels[:subset_size]

def run_eda():
  # EDA - Generate reports
  generate_eda_report(train_df)
  pass

def run_feature_engineering():
  # Feature Engineering & Explainability
  X_train, X_val, y_train, y_val, model = feature_engineering(train_df)
  pass
  
def run_automl():
  # AutoML & Hyperparameter Optimization
  best_params = automl_optimization(X_train, y_train, X_val, y_val)

def _run_monitoring():
  # Model Monitoring and Drift Detection
  model_monitoring(model, X_val, y_val)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "eda":
            run_eda()
        elif sys.argv[1] == "feature_engineering":
            run_feature_engineering()
        elif sys.arg[1] == "automl":
            run_automl()
        elif sys.arg[1] == "monitoring":
            run_monitoring()
