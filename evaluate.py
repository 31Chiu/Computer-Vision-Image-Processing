import cv2
import numpy as np
import os
from skimage import feature
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths
current_directory = os.getcwd()
image_folder_path = os.path.join(current_directory, "Validation")
trained_model_path = "gender_classification_svm.pkl"
class_file_path = "classes.txt"

# Load class names
with open(class_file_path, "r") as file:
    class_names = [row.strip() for row in file.readlines()]

def extract_hog_features(image):
    """Extract HOG features from an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = feature.hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

def load_images_and_labels(data_dir, target_size=(128, 128)):
    """Load images and labels from a directory, resize images, and extract HOG features."""
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, target_size)
                    hog_features = extract_hog_features(image)
                    features.append(hog_features)
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
            conf_matrix.T, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix For Gender Classification Model")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()