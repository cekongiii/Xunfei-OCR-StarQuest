import cv2
import numpy as np

def order_points(pts):
    """
    对四个点进行排序：左上 -> 右上 -> 右下 -> 左下
    解决算法找出的角点顺序不固定的问题
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 1. 找左上(0)和右下(2)
    # 左上角的 x+y 和最小，右下角的 x+y 和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 2. 找右上(1)和左下(3)
    # 右上角的 y-x 最小 (diff)，左下角的 y-x 最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_doc_contours(img_path):
    """
    核心流程：读取 -> 灰度 -> 高斯模糊 -> Canny边缘 -> 轮廓近似
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 错误：无法读取图片 {img_path}")
        return None, None
        
    # 1. 预处理：去噪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 边缘检测 (阈值可根据实际光照调整)
    edged = cv2.Canny(blur, 75, 200)
    
    # 3. 寻找轮廓
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积从大到小排序，只看前5个最大的
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screen_cnt = None
    
    for c in cnts:
        # 计算周长
        peri = cv2.arcLength(c, True)
        # 多边形逼近：0.02 是精度系数，越小拟合越紧密
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 如果逼近后刚好是 4 个点，说明找到了纸张边缘
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    return img, screen_cnt

def auto_scan(img_path, output_path):
    print(f"🔄 正在处理: {img_path} ...")
    
    # 1. 获取原图和角点
    image, screen_cnt = get_doc_contours(img_path)
    
    if screen_cnt is None:
        print("⚠️ 未检测到文档边缘，跳过矫正。")
        return

    print(f"✅ 检测到边缘，角点坐标:\n{screen_cnt.reshape(4,2)}")

    # 2. 准备透视变换
    pts = screen_cnt.reshape(4, 2)
    rect = order_points(pts) # 关键：排序
    
    # 定义目标尺寸 (模拟 A4 纸比例 3:4)
    w, h = 300, 400
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 3. 计算变换矩阵并拉直
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    
    # 4. 保存结果
    cv2.imwrite(output_path, warped)
    print(f"🚀 矫正完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    # 使用昨天生成的模拟歪斜图进行测试
    auto_scan("2_camera_slanted.png", "auto_scan_result.png")