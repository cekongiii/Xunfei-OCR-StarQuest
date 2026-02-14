import cv2
import numpy as np

# 1.读取图像
image = cv2.imread("star_python.png")

# 2.转化为灰度图
gray1 = cv2.imread("star_python.png", cv2.IMREAD_GRAYSCALE)
# 2.(1). 先转成浮点数进行计算
noisy_gray = gray1.astype(np.float32) + np.random.normal(0, 30, gray1.shape)

# 2.(2). 限制范围在 0-255 之间，防止越界
noisy_gray = np.clip(noisy_gray, 0, 255)

# 2.(3). 再转回 uint8 供 OpenCV 处理
gray1 = noisy_gray.astype(np.uint8)

# 3.阈值化
ret,thresh = cv2.threshold(gray1,80,255,cv2.THRESH_BINARY)       
# cv2.threshold 会返回两个值：一个是阈值，一个是处理后的图像
# 我们通常用 _ 忽略第一个值，把图存进 thresh

# 4.不能用imshow 用imwrite保存结果
cv2.imwrite("result_binary.png",thresh)

print("✅ 处理完成！结果已保存为 'result_binary.png'，请在左侧文件栏双击查看。")

# 在 WSL 里，这些 GUI 相关的代码可以注释掉或删掉
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#WSL 是一个运行在 Windows 里的 Linux“大脑”，它默认是没有“脸”（图形界面）的。
#cv2.imshow 试图弹出一个窗口来显示图片，但 Linux 找不到显示器，于是它就罢工了。
#知识补丁：在服务器开发、云端算法部署或者 WSL 环境中，我们通常不用 imshow，而是采用 “写文件法”。

# 5. 寻找轮廓
# cv2.RETR_EXTERNAL: 只找最外层的轮廓（不找洞中洞）
# cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，只保留端点（节省内存）
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"📊 系统检测到的目标总数: {len(contours)}")

# 循环每一个轮廓，打印出它的面积
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 2: # 简单的面积过滤，对应毕设里的去噪思路
        print(f"🌟 发现第 {i} 颗星: 面积 = {area}")