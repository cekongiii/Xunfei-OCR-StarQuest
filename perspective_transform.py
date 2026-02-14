import cv2
import numpy as np

# --- 第一阶段：制造一张完美的“原件” ---
width, height = 300, 400
# 白纸
original_doc = np.ones((height, width, 3), dtype=np.uint8) * 255 

# 在纸上写正正方方的字
cv2.putText(original_doc, "XunFei", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
cv2.putText(original_doc, "OCR", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
cv2.rectangle(original_doc, (10, 10), (width-10, height-10), (0, 255, 0), 3) # 画个框方便看边

# 保存第一张图：完美的纸
cv2.imwrite("1_perfect_doc.png", original_doc)


# --- 第二阶段：模拟相机拍摄（把它变歪） ---

# 准备一张大的黑色背景（模拟桌子）
background = np.zeros((600, 600, 3), dtype=np.uint8)

# 定义完美的坐标 (源)
pts_flat = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# 定义歪斜的坐标 (目标：模拟相机看到的梯形)
pts_slanted = np.float32([[150, 150], [400, 100], [450, 400], [50, 450]])

# 计算变换矩阵 M1 (从正 -> 歪)
M_simulate = cv2.getPerspectiveTransform(pts_flat, pts_slanted)

# 执行变换，把白纸“贴”到黑色背景上
# warpPerspective 不仅能把图变歪，还能把它放到指定位置
slanted_img = cv2.warpPerspective(original_doc, M_simulate, (600, 600))

# 保存第二张图：相机拍到的歪图
cv2.imwrite("2_camera_slanted.png", slanted_img)


# --- 第三阶段：OCR 算法矫正（把你刚才学的逻辑用在这里） ---

# 现在的任务是：把 slanted_img 变回 flat
# 源点：就是刚才那个歪斜的坐标 pts_slanted
# 目标点：我们要把它拉回的大小 (300x400)

pts_src = pts_slanted # 刚才的目标变成了现在的源
pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) # 拉回正方形

# 计算变换矩阵 M2 (从歪 -> 正)
M_restore = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 执行变换
restored_img = cv2.warpPerspective(slanted_img, M_restore, (width, height))

# 保存第三张图：矫正后的结果
cv2.imwrite("3_ocr_restored.png", restored_img)

print("✅ 仿真完成！请依次查看文件夹中的三张图片：")
print("1. 1_perfect_doc.png -> 原始文件")
print("2. 2_camera_slanted.png -> 模拟拍摄（字是歪的）")
print("3. 3_ocr_restored.png -> 算法矫正（字变正了！）")