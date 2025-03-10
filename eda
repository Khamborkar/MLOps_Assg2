import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import sweetviz as sv
from tensorflow.keras.datasets import fashion_mnist

import torchvision
import torch

# Download Fashion MNIST
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Save the dataset as a .tar file or other format for DVC tracking


# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert to Pandas DataFrame
train_df = pd.DataFrame(train_images.reshape(-1, 28 * 28))  # Flattening images
train_df['label'] = train_labels  # Add labels

# Generate Pandas Profiling report
profile = ProfileReport(train_df, explorative=True)
profile.to_file("reports/fashion_mnist_eda.html")

# Generate Sweetviz report
sweet_report = sv.analyze(train_df)
sweet_report.show_html("reports/fashion_mnist_sweetviz.html")

# Plot class distribution
sns.countplot(x=train_labels)
plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("reports/class_distribution.png")
plt.show()

print("EDA reports generated successfully!")
