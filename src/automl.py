# src/automl.py
from tpot import TPOTClassifier
import optuna
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
