import os
import tarfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.transforms import ToPILImage
from ydata_profiling import ProfileReport
import sweetviz as sv
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert dataset to Pandas DataFrame (subset for efficiency)
subset_size = 500  
train_df = pd.DataFrame(train_images[:subset_size].reshape(subset_size, -1))
train_df['label'] = train_labels[:subset_size]


# #------------------------------------------------------------------------------------------
# # Feature Engineering & Explainability - SHAP visualization and preprocessing
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(train_df.drop('label', axis=1))  # Scale features
# y = train_df['label']
# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(X_scaled)
# X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # Random Forest model for explainability
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_val)
# shap.summary_plot(shap_values, X_val)
# plt.savefig("reports/shap_summary_plot.png")
# plt.show()

# # AutoML & Hyperparameter Optimization
# tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20)
# tpot.fit(X_train, y_train)
# tpot.export('best_model.py')

# def objective(trial):
#     max_depth = trial.suggest_int('max_depth', 5, 20)
#     n_estimators = trial.suggest_int('n_estimators', 50, 200)
#     model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
#     model.fit(X_train, y_train)
#     return model.score(X_val, y_val)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)
# print("Best hyperparameters:", study.best_params)

# # MLflow - Model Monitoring and Drift Detection
# with mlflow.start_run():
#     mlflow.log_param("model_type", "RandomForest")
#     mlflow.log_param("max_depth", 10)
#     mlflow.log_param("n_estimators", 100)
#     mlflow.sklearn.log_model(model, "model")
#     mlflow.log_metric("accuracy", model.score(X_val, y_val))

# # Drift detection (simple version)
# initial_accuracy = model.score(X_val, y_val)
# new_accuracy = accuracy_score(y_val, model.predict(X_val))  # Simulate new data
# if initial_accuracy - new_accuracy > 0.05:
#     print("Model performance has drifted. Retraining is recommended.")
# else:
#     print("No significant drift detected.")
# #-------------------------------------------------------------------------------------------

def generate_eda_report(train_df):
    # Function to save dataset as tar file
    def save_as_tar(dataset, file_name, batch_size=100):
        os.makedirs("data", exist_ok=True)  # Ensure the directory exists
        to_pil = ToPILImage()
        
        with tarfile.open(file_name, "w:gz") as tar:
            image_paths = []
            
            for i, (img, label) in enumerate(dataset):
                img_path = f"data/{i}_{label}.png"
                to_pil(img).save(img_path)  # Save as PNG
                image_paths.append(img_path)
                
                if len(image_paths) >= batch_size:  # Batch processing
                    for path in image_paths:
                        tar.add(path, arcname=os.path.basename(path))
                        os.remove(path)  # Remove after adding
                    image_paths.clear()
    
            # Add and remove remaining images
            for path in image_paths:
                tar.add(path, arcname=os.path.basename(path))
                os.remove(path)
    
    # Download and save Fashion MNIST
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    
    
    save_as_tar(train_data, 'fashion_mnist_train.tar.gz')
    save_as_tar(test_data, 'fashion_mnist_test.tar.gz')
    
    # Generate Pandas Profiling report (optimized settings)
    profile = ProfileReport(train_df, minimal=True)
    profile.to_file("reports/fashion_mnist_eda.html")
    
    # # Generate Sweetviz report
    # sv.analyze(train_df).show_html("reports/fashion_mnist_sweetviz.html")
    # Generate Sweetviz report with pairwise analysis explicitly turned off
    sweet_report = sv.analyze(train_df, pairwise_analysis="off")
    sweet_report.show_html("reports/fashion_mnist_sweetviz.html")
    
    # Plot class distribution
    sns.countplot(x=train_labels[:subset_size])
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/class_distribution.png")
    plt.show()
    
    print("EDA reports generated successfully!")

# Call the function from the EDA script to generate reports
generate_eda_report(train_df)
