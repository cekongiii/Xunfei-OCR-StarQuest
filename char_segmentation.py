import cv2
import numpy as np
import os

def get_projection(binary_img, axis=0):
    """
    计算投影直方图
    axis=0: 垂直投影 (统计每一列的白点) -> 用来切字符
    axis=1: 水平投影 (统计每一行的白点) -> 用来切行
    """
    # 归一化：将像素总和除以255，得到“白色像素的个数”
    projection = np.sum(binary_img, axis=axis) / 255
    return projection

def get_cuts(projection, threshold=5, min_size=5):
    """
    通用的切割算法：根据投影和阈值，找出所有的 [start, end] 区间
    """
    cuts = []
    start = 0
    in_block = False
    
    for i, val in enumerate(projection):
        if val > threshold:
            if not in_block:
                in_block = True
                start = i
        else:
            if in_block:
                in_block = False
                end = i
                if (end - start) > min_size:
                    cuts.append((start, end))
                    
    return cuts

def ocr_pipeline(img_path):
    print(f"🚀 开始处理: {img_path}")
    
    # 1. 读入与预处理
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 图片未找到")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 【关键修正】：白纸黑字 -> 必须用 INV -> 变成 黑纸白字
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 【关键修正】：屏蔽绿色边框 (Masking)
    # 强制把四周 10 像素涂黑，消灭那个绿框带来的 400 高度信号
    h, w = binary.shape
    margin = 20
    binary[0:margin, :] = 0
    binary[h-margin:h, :] = 0
    binary[:, 0:margin] = 0
    binary[:, w-margin:w] = 0
    
    cv2.imwrite("debug_binary_final.png", binary)
    print("✅ 预处理完成，已去除绿边。")

    # ==========================================
    # 第一步：水平投影，切分“行” (Line Segmentation)
    # ==========================================
    h_proj = get_projection(binary, axis=1) # axis=1 是横向
    line_cuts = get_cuts(h_proj, threshold=5, min_size=10)
    
    print(f"📋 检测到 {len(line_cuts)} 行文字")
    
    output_dir = "ocr_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每一行
    for i, (y_start, y_end) in enumerate(line_cuts):
        # 稍微放宽一点边界
        y_start = max(0, y_start - 2)
        y_end = min(h, y_end + 2)
        
        # 把这一行切出来 (注意是切 binary 还是原图？通常切 binary 方便后续处理，或者切原图做展示)
        # 这里我们切 binary 来做下一步分析，切原图用来保存
        line_binary = binary[y_start:y_end, :]
        line_img = img[y_start:y_end, :] 
        
        # 保存行图片
        cv2.imwrite(f"{output_dir}/line_{i}.png", line_img)
        print(f"  -> 处理第 {i+1} 行 (高度 {y_end-y_start}px)...")

        # ==========================================
        # 第二步：垂直投影，切分“字” (Char Segmentation)
        # ==========================================
        v_proj = get_projection(line_binary, axis=0) # axis=0 是纵向
        char_cuts = get_cuts(v_proj, threshold=2, min_size=5)
        
        print(f"     检测到 {len(char_cuts)} 个字符")
        
        for j, (x_start, x_end) in enumerate(char_cuts):
            # 切割单个字符
            x_start = max(0, x_start - 2)
            x_end = min(w, x_end + 2)
            
            char_roi = line_img[:, x_start:x_end]
            
            # 保存：文件名格式 line_行号_char_字号.png
            filename = f"{output_dir}/line_{i}_char_{j}.png"
            cv2.imwrite(filename, char_roi)

    print(f"🎉 全部完成！结果保存在 {output_dir} 文件夹中。")

if __name__ == "__main__":
    ocr_pipeline("auto_scan_result.png")