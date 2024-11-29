import cv2
import numpy as np
import os
from skimage import feature
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import joblib

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = feature.hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

def load_images_and_labels(data_dir, target_size=(128, 128)):
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

# Load dataset
data_dir = './Training/'
# data_dir = './Temp_Training/'
features, labels = load_images_and_labels(data_dir)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM model
model = svm.SVC(probability=True)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'train_gender_classification_svm.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_precision = precision_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro Precision: {macro_precision}")

# Ensure the model meets the performance criteria
if accuracy >= 0.7 and macro_recall >= 0.7 and macro_precision >= 0.7:
    print("Model meets the performance criteria.")
else:
    print("Model does not meet the performance criteria.")