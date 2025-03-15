# src/feature_engineering.py
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import tensorflow
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert dataset to Pandas DataFrame (subset for efficiency)
subset_size = 500  
train_df = pd.DataFrame(train_images[:subset_size].reshape(subset_size, -1))
train_df['label'] = train_labels[:subset_size]

# Feature Engineering & Explainability function
def feature_engineering(train_df):
    # Preprocessing - Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df.drop('label', axis=1))  # Scale features
    y = train_df['label']
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Explainability - SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # SHAP summary plot
    shap.summary_plot(shap_values, X_val)
    plt.savefig("reports/shap_summary_plot.png")
    plt.show()

    print("Feature Engineering and Explainability completed!")

    return X_train, X_val, y_train, y_val, model
