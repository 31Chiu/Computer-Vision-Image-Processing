import cv2
import numpy as np
import os
import glob
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

def classify_image(image_path, model):
    features = extract_resnet_features(image_path)
    if features is None:
        return None, None
    prediction = model.predict([features])
    confidence_scores = model.predict_proba([features])
    predicted_class_index = model.classes_.tolist().index(prediction[0])
    confidence_scores = confidence_scores[0][predicted_class_index]
    return prediction[0], confidence_scores

def main():
    # 加载训练好的 SVM 模型
    model = joblib.load('gender_classification_svm.pkl')

    while True:
        folder_name = input("\nPlease enter the name of the folder containing images you want to classify: ")
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

if __name__ == "__main__":
    main()