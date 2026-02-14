import cv2
import numpy as np

# 1. 造星（还是那个配方，模拟一颗位于 (100, 100) 的星）
img = np.zeros((200, 200), dtype=np.uint8)
img[98:103, 98:103] = 255 # 画个亮块
img = cv2.GaussianBlur(img, (5, 5), 1.5) # 模糊一下

# 2. 粗定位：二值化 + 找轮廓
_, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"检测到 {len(contours)} 个候选目标")

for i, cnt in enumerate(contours):
    # --- 关键步骤 A: 获取边界框 (Bounding Rect) ---
    # x, y 是左上角坐标；w, h 是宽和高
    x, y, w, h = cv2.boundingRect(cnt)
    
    # --- 关键步骤 B: 提取 ROI (切片) ---
    # 注意：numpy 切片是 img[行(y):行结尾, 列(x):列结尾]
    roi = img[y : y+h, x : x+w]
    
    # --- 关键步骤 C: 在小窗口内计算矩 ---
    M = cv2.moments(roi)
    
    if M["m00"] != 0:
        # 算出的是相对于 roi 左上角的坐标
        cx_local = M["m10"] / M["m00"]
        cy_local = M["m01"] / M["m00"]
        
        # --- 关键步骤 D: 坐标还原 (Global Coordinates) ---
        # 全局坐标 = 局部坐标 + 窗口左上角偏移
        cx_global = cx_local + x
        cy_global = cy_local + y
        
        print(f"🌟 星星 #{i+1}:")
        print(f"   - 边界框: x={x}, y={y}, w={w}, h={h}")
        print(f"   - 局部质心: ({cx_local:.2f}, {cy_local:.2f})")
        print(f"   - 全局精测坐标: ({cx_global:.2f}, {cy_global:.2f})")
        
        # 可视化：在原图上画框
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, 1)

# 保存看看框画得对不对
cv2.imwrite("star_roi_result.png", img)