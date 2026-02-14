import cv2
import numpy as np
import os

def preprocess_chars(char_dir):
    processed_data = []
    
    # 获取所有切好的字符图
    char_files = [f for f in os.listdir(char_dir) if f.endswith('.png')]
    
    for f in char_files:
        path = os.path.join(char_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 必须是灰度图
        
        # 1. 缩放到 28x28
        img_resized = cv2.resize(img, (28, 28))
        
        # 2. 归一化 (Normalization)
        # 像素值从 0-255 变成 0-1 之间，这能让神经网络收敛得更快
        img_normalized = img_resized.astype('float32') / 255.0
        
        # 3. 增加一个维度（通道维）
        # CNN 要求输入是 (高度, 宽度, 通道数)
        img_final = np.expand_dims(img_normalized, axis=-1)
        
        processed_data.append(img_final)
        
    return np.array(processed_data)

if __name__ == "__main__":
    data = preprocess_chars("ocr_result")
    print(f"✅ 成功处理 {len(data)} 个字符，数据形状: {data.shape}")