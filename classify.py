import cv2
import numpy as np
from skimage import feature
import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import glob
import os

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = feature.hog(gray_image, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False)
    return hog_features

def load_images_and_labels(image_paths, labels):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            hog_features = extract_hog_features(image)
            features.append(hog_features)
        else:
            print(f"Warning: Image at path {image_path} could not be loaded.")
    return np.array(features), np.array(labels)

def classify_image(image_path, model, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Image not found: {image_path}')
        return None, None
    image = cv2.resize(image, target_size)
    hog_features = extract_hog_features(image)

    prediction = model.predict([hog_features])
    confidence_scores = model.predict_proba([hog_features])

    predicted_class_index = model.classes_.tolist().index(prediction[0])
    confidence_scores = confidence_scores[0][predicted_class_index]

    return prediction[0], confidence_scores

# Load dataset
male_image_paths = glob.glob('./Validation/male/*.[jp][pn]g')[:100]
female_image_paths = glob.glob('./Validation/female/*.[jp][pn]g')[:100]
image_paths = male_image_paths + female_image_paths
labels = ['male'] * len(male_image_paths) + ['female'] * len(female_image_paths)

# Debugging information
print(f"Number of male images: {len(male_image_paths)}")
print(f"Number of female images: {len(female_image_paths)}")
print(f"Total number of images: {len(image_paths)}\n")

# Extract features and labels
features, labels = load_images_and_labels(image_paths, labels)

# Debugging information
print(f"Number of features extracted: {len(features)}")
print(f"Number of labels: {len(labels)}\n")

# Check if features and labels are not empty
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No features or labels were loaded. Please check the image paths and ensure images are available.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM model
model = svm.SVC(probability=True)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'gender_classification_svm.pkl')

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

# Classify images in a user-specified folder
while True:
    folder_name = input("\nPlease enter the name of the folder containing images you want to classify: ")
    # folder_path = os.path.join('./data_test/', folder_name)
    folder_path = os.path.join('./Validation/', folder_name)
    image_paths = glob.glob(os.path.join(folder_path, '*.[jp][pn]g'))

    if not image_paths:
        print("No images found in the specified folder. Please try again.")
    else:
        for image_path in image_paths:
            result, confidence = classify_image(image_path, model)
            if result is not None and confidence is not None:
                print(f"Image: {image_path}")
                print(f"Prediction: {result}")
                print(f"Confidence Scores: {confidence}\n")
            else:
                print(f"Failed to classify the image: {image_path}\n")
        break
