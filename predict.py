import cv2
import numpy as np
import tensorflow as tf

def predict_char(img_path):
    # 1. 加载模型
    model = tf.keras.models.load_model('letter_ocr_model.h5')
    
    # 2. 读取并预处理
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 没找到图！")
        return
        
    # 缩放
    img_resized = cv2.resize(img, (28, 28))
    
    # 【非常关键】归一化并调整为模型需要的格式
    img_input = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # 3. 推理
    prediction = model.predict(img_input)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # 4. 映射回字母
    # EMNIST 1=A, 2=B... 所以 chr(64 + 1) = 'A'
    if result_index > 0:
        char = chr(64 + result_index)
    else:
        char = "Unknown"
        
    print(f"\n" + "="*30)
    print(f"🖼️ 测试图片: {img_path}")
    print(f"🧠 AI 识别结果: 【 {char} 】")
    print(f"📊 信心指数: {confidence*100:.2f}%")
    print(f"="*30)

if __name__ == "__main__":
    # 试试昨天那个被误认为是 "2" 的 "X"！
    predict_char("ocr_result/line_0_char_0.png")