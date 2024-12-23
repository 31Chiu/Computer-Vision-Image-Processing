import cv2
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Paths
current_directory = os.getcwd()
image_folder_path = os.path.join(current_directory, "Validation")
trained_model_path = "gender_classification_svm.pkl"
class_file_path = "classes.txt"

# Load class names
with open(class_file_path, "r") as file:
    class_names = [row.strip() for row in file.readlines()]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet18 model
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Identity()  # Remove the final fully connected layer

# Move model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18.to(device)
resnet18.eval()

def extract_resnet_features(image_path):
    """Extract features from an image using pre-trained ResNet18 model."""
    image = cv2.imread(image_path)
    if image is None:
        print(f'Image not found: {image_path}')
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)  # Convert NumPy array to PIL image
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet18(image).cpu().numpy().flatten()
    return features

def load_images_and_labels(data_dir):
    """Load images and labels from a directory and extract ResNet18 features."""
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                features_vector = extract_resnet_features(image_path)
                if features_vector is not None:
                    features.append(features_vector)
                    labels.append(label)
    return np.array(features), np.array(labels)

def main():
    try:
        # Load dataset
        features, labels = load_images_and_labels(image_folder_path)

        # Load the trained SVC model
        svc_model = joblib.load(trained_model_path)

        # Make predictions
        predictions = svc_model.predict(features)

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        # plt.xlabel("True")
        # plt.ylabel("Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix For Gender Classification Model")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()