import cv2
import numpy as np
import tensorflow as tf
import os
import re # 别忘了在开头导入 re 模块

def run_full_ocr(folder_path, model_path):
    model = tf.keras.models.load_model(model_path)
    
    # 1. 定义一个更强悍的解析器
    def get_info(filename):
        # 使用正则表达式匹配数字：寻找 line_数字_char_数字
        match = re.search(r'line_(\d+)_char_(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    # 2. 过滤并排序
    all_files = os.listdir(folder_path)
    valid_files_info = []
    
    for f in all_files:
        info = get_info(f)
        if info:
            valid_files_info.append((info[0], info[1], f))
    
    # 先按行排，再按字符序号排
    valid_files_info.sort()

    print(f"📂 发现 {len(valid_files_info)} 个标准字符切片，开始识别...\n")

    current_line = -1
    full_text = ""
    line_text = ""

    for line_idx, char_idx, filename in valid_files_info:
        # 初始化第一行
        if current_line == -1: current_line = line_idx
        
        # 换行处理
        if line_idx != current_line:
            print(f"第 {current_line} 行识别结果: {line_text}")
            full_text += line_text + "\n"
            line_text = ""
            current_line = line_idx

        # 推理逻辑
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))
        img_input = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        prediction = model.predict(img_input, verbose=0)
        result_idx = np.argmax(prediction)
        
        # 映射 A-Z
        char = chr(64 + result_idx) if result_idx > 0 else "?"
        line_text += char

    # 打印最后一行
    if line_text:
        print(f"第 {current_line} 行识别结果: {line_text}")
        full_text += line_text

    print("\n" + "="*30)
    print("📜 最终识别全文：")
    print(full_text)
    print("="*30)

if __name__ == "__main__":
    run_full_ocr("ocr_result", "letter_ocr_model.h5")