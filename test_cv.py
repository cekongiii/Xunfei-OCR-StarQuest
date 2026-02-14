import cv2
import numpy as np

# 1. 创建一个 512x512 的黑色画布 (对应 MATLAB 的 zeros)
# np.uint8 对应 C 语言的 unsigned char，也就是灰度图的 0-255
img = np.zeros((512, 512), dtype=np.uint8)

# 2. 点亮星星 (对应 MATLAB 的 img(99:101, 99:101) = 200)
# 注意：Python 的切片是左闭右开 [开始:结束]
img[99:102, 99:102] = 200

# 3. 保存图像
cv2.imwrite('star_python.png', img)

print("图像已生成！用 explorer.exe . 打开文件夹看看？")