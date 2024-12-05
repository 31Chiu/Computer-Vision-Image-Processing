import cv2
import numpy as np
import os
import glob
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

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

def upload_images():
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        for widget in results_frame.winfo_children():
            widget.destroy()  # 清除之前的结果
        for i, file_path in enumerate(file_paths):
            result, confidence = classify_image(file_path, model)
            if result is not None and confidence is not None:
                img = Image.open(file_path)
                img = img.resize((224, 224), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                
                # 创建一个框架来包含图像和结果
                frame = tk.Frame(results_frame, bd=2, relief="solid")
                frame.grid(row=i//5, column=i%5, padx=10, pady=10, sticky="nsew")

                img_label = tk.Label(frame, image=img)
                img_label.image = img  # 保持对图像的引用
                img_label.pack(side="top", pady=10)

                result_text = f"Prediction: {result}\nConfidence Scores: {confidence:.2f}"
                result_label = tk.Label(frame, text=result_text, justify="center")
                result_label.pack(side="top", pady=10)
            else:
                messagebox.showerror("Error", f"Failed to classify the image: {file_path}")

def main():
    global model, results_frame, root

    # 加载训练好的 SVM 模型
    model = joblib.load('gender_classification_svm.pkl')

    # 创建主窗口
    root = tk.Tk()
    root.title("Gender Classifier")

    # 创建一个居中的框架
    main_frame = tk.Frame(root)
    main_frame.pack(pady=20, padx=20, fill="both", expand=True)

    # 创建上传按钮
    upload_btn = tk.Button(main_frame, text="Upload Images", command=upload_images)
    upload_btn.pack(pady=20)

    # 创建显示结果的框架和滚动条
    canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    results_frame = tk.Frame(canvas)

    results_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=results_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # 运行主循环
    root.mainloop()

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

if __name__ == "__main__":
    main()