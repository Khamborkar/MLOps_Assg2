# src/eda.py
import pandas as pd
from ydata_profiling import ProfileReport
import sweetviz as sv
import numpy
# import main.py

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Convert dataset to Pandas DataFrame (subset for efficiency)
subset_size = 500  
train_df = pd.DataFrame(train_images[:subset_size].reshape(subset_size, -1))
train_df['label'] = train_labels[:subset_size]

# EDA function that accepts DataFrame
def generate_eda_report(train_df):
    import main
    
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


