import cv2
import numpy as np
import os
import glob
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from PIL import Image  # 导入 PIL 库

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的 ResNet18 模型
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Identity()  # 移除最后的全连接层

# 将模型移动到设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18.to(device)
resnet18.eval()

def extract_resnet_features(image_path):
    """使用预训练的 ResNet18 模型提取图像特征。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f'Image not found: {image_path}')
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)  # 将 NumPy 数组转换为 PIL 图像
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet18(image).cpu().numpy().flatten()
    return features

def load_images_and_labels(image_paths, labels):
    features = []
    valid_labels = []
    for image_path, label in zip(image_paths, labels):
        features_vector = extract_resnet_features(image_path)
        if features_vector is not None:
            features.append(features_vector)
            valid_labels.append(label)
        else:
            print(f"Warning: Image at path {image_path} could not be loaded.")
    return np.array(features), np.array(valid_labels)

def classify_image(image_path, model, target_size=(224, 224)):
    features = extract_resnet_features(image_path)
    if features is None:
        return None, None
    prediction = model.predict([features])
    confidence_scores = model.predict_proba([features])
    predicted_class_index = model.classes_.tolist().index(prediction[0])
    confidence_scores = confidence_scores[0][predicted_class_index]
    return prediction[0], confidence_scores

# 加载数据集
male_image_paths = glob.glob('./Temp_Training/Temp_male/*.[jp][pn]g')[:100]
female_image_paths = glob.glob('./Temp_Training/Temp_female/*.[jp][pn]g')[:100]
image_paths = male_image_paths + female_image_paths
labels = ['male'] * len(male_image_paths) + ['female'] * len(female_image_paths)

# 调试信息
print(f"Number of male images: {len(male_image_paths)}")
print(f"Number of female images: {len(female_image_paths)}")
print(f"Total number of images: {len(image_paths)}\n")

# 提取特征和标签
features, labels = load_images_and_labels(image_paths, labels)

# 调试信息
print(f"Number of features extracted: {len(features)}")
print(f"Number of labels: {len(labels)}\n")

# 检查特征和标签是否为空
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No features or labels were loaded. Please check the image paths and ensure images are available.")

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 训练 SVM 模型
model = svm.SVC(probability=True)
model.fit(X_train, y_train)

# 保存训练好的模型
joblib.dump(model, 'gender_classification_svm.pkl')

# 评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_precision = precision_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro Precision: {macro_precision}")

# 确保模型满足性能标准
if accuracy >= 0.7 and macro_recall >= 0.7 and macro_precision >= 0.7:
    print("Model meets the performance criteria.")
else:
    print("Model does not meet the performance criteria.")

# # 分类用户指定文件夹中的图像
# while True:
#     folder_name = input("\nPlease enter the name of the folder containing images you want to classify: ")
#     folder_path = os.path.join('./Validation/', folder_name)
#     image_paths = glob.glob(os.path.join(folder_path, '*.[jp][pn]g'))

#     if not image_paths:
#         print("No images found in the specified folder. Please try again.")
#     else:
#         for image_path in image_paths:
#             result, confidence = classify_image(image_path, model)
#             if result is not None and confidence is not None:
#                 print(f"Image: {image_path}")
#                 print(f"Prediction: {result}")
#                 print(f"Confidence Scores: {confidence}\n")
#             else:
#                 print(f"Failed to classify the image: {image_path}\n")
#         break