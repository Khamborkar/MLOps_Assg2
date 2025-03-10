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
from torchvision.transforms import ToPILImage

# Download Fashion MNIST
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Save the dataset as a tar file in batches
def save_as_tar(dataset, file_name):
    to_pil = ToPILImage()  # Initialize ToPILImage transform
    image_paths = []  # To store image file paths for batching
    with tarfile.open(file_name, "w:gz") as tar:
        for i, (img, label) in enumerate(dataset):
            img = to_pil(img)  # Convert the tensor to a PIL Image
            img_path = os.path.join('data', f"{i}_{label}.png")  # Save each image with an index and label
            img.save(img_path)
            image_paths.append(img_path)  # Add path to the list
            
            if len(image_paths) >= 100:  # Save in batches of 100 images
                for path in image_paths:
                    tar.add(path, arcname=os.path.basename(path))  # Add image to the tar file
                image_paths.clear()  # Clear paths list after adding to the tar file

        # Add remaining images if any
        for path in image_paths:
            tar.add(path, arcname=os.path.basename(path))

        # Clean up image files after adding them to the tar
        for path in image_paths:
            os.remove(path)

# Save training and test data
save_as_tar(train_data, 'fashion_mnist_train.tar.gz')
save_as_tar(test_data, 'fashion_mnist_test.tar.gz')

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert to Pandas DataFrame (only a subset for profiling)
subset_size = 500  # Only use the first 1000 images for profiling
train_df = pd.DataFrame(train_images[:subset_size].reshape(-1, 28 * 28))  # Flattening images
train_df['label'] = train_labels[:subset_size]  # Add labels

# Generate Pandas Profiling report
# profile = ProfileReport(train_df, explorative=True)
# profile = ProfileReport(train_df.head(1000), explorative=True)
profile = ProfileReport(train_df.head(1000), explorative=True, config={'correlations': {'auto': False}})
profile.to_file("reports/fashion_mnist_eda.html")

# Generate Sweetviz report
sweet_report = sv.analyze(train_df)
sweet_report.show_html("reports/fashion_mnist_sweetviz.html")

# Plot class distribution
sns.countplot(x=train_labels[:subset_size])  # Plot class distribution for the subset
plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("reports/class_distribution.png")
plt.show()

print("EDA reports generated successfully!")
