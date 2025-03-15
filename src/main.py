# main.py
import pandas as pd
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

# EDA - Generate reports
generate_eda_report(train_df)

# Feature Engineering & Explainability
X_train, X_val, y_train, y_val, model = feature_engineering(train_df)

# AutoML & Hyperparameter Optimization
best_params = automl_optimization(X_train, y_train, X_val, y_val)

# Model Monitoring and Drift Detection
model_monitoring(model, X_val, y_val)
