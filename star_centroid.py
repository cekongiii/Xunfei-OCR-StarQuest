import cv2
import numpy as np

# 1. 模拟 MATLAB 的造星过程
img = np.zeros((512, 512), dtype=np.uint8)
img[100:103, 100:103] = 255  # 亮一点，大一点
# 模拟高斯模糊（让质心产生亚像素偏移）
img = cv2.GaussianBlur(img, (7, 7), 1.5)

# 2. 阈值化提取
_, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

# 3. 寻找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # 计算图像矩
    M = cv2.moments(cnt)
    
    # 防止分母为 0
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        print(f"🌟 发现星点！亚像素质心坐标: ({cX:.2f}, {cY:.2f})")