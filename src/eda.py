
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import sweetviz as sv
from tensorflow.keras.datasets import fashion_mnist

import torchvision
import torch
import tarfile
import os

# Download Fashion MNIST
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Save the dataset as a .tar file or other format for DVC tracking
# Save the dataset as a tar file
def save_as_tar(dataset, file_name):
    with tarfile.open(file_name, "w:gz") as tar:
        for data_item in dataset:
            img, label = data_item
            img_path = os.path.join('data', f"{label}.png")  # Save each image
            img.save(img_path)
            tar.add(img_path, arcname=os.path.basename(img_path))  # Add the image to the tar file

# Save training and test data
save_as_tar(train_data, 'fashion_mnist_train.tar.gz')
save_as_tar(test_data, 'fashion_mnist_test.tar.gz')

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
