# src/automl.py
from tpot import TPOTClassifier
import optuna
import tensorflow
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert dataset to Pandas DataFrame (subset for efficiency)
subset_size = 500  
train_df = pd.DataFrame(train_images[:subset_size].reshape(subset_size, -1))
train_df['label'] = train_labels[:subset_size]

# AutoML & Hyperparameter Optimization function
def automl_optimization(X_train, y_train, X_val, y_val):
    # AutoML using TPOT
    tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20)
    tpot.fit(X_train, y_train)
    tpot.export('best_model.py')
    print("AutoML Model Exported.")

    # Hyperparameter optimization using Optuna
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 20)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    # Run the optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best hyperparameters:", study.best_params)
    
    return study.best_params
