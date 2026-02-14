import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. 专门解析 idx-ubyte 格式的函数
def load_emnist_ubyte(folder_path):
    def read_labels(filename):
        # 既然没有 .gz，我们直接用普通的 open 读取二进制 ('rb')
        with open(filename, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    def read_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

    print("⏳ 正在根据本地文件名解析二进制文件...")
    
    # 【请根据你 ls 的结果对齐下面四个名字】
    # 如果你的文件名是 emnist-letters-train-images-idx3-ubyte (没有.gz)
    train_images_path = os.path.join(folder_path, 'emnist-letters-train-images-idx3-ubyte')
    train_labels_path = os.path.join(folder_path, 'emnist-letters-train-labels-idx1-ubyte')
    test_images_path = os.path.join(folder_path, 'emnist-letters-test-images-idx3-ubyte')
    test_labels_path = os.path.join(folder_path, 'emnist-letters-test-labels-idx1-ubyte')

    # 检查一下文件到底在不在，不在就报错提醒
    for p in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ 还是没找到文件: {p}\n请检查文件名是否带后缀，或者中间是横杠还是下划线！")

    x_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path)
    x_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)

    # 预处理：归一化并调整形状
    # 原始 IDX 格式通常需要旋转 90 度并镜像翻转才能变正
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    
    # 这一步是修正 EMNIST 常见的倒置问题
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"✅ 加载成功！训练集形状: {x_train.shape}")
    return x_train, y_train, x_test, y_test
    
def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(27, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # 指向你解压后的那个文件夹路径
    data_folder = "/home/liuby/projects/datasets/emnist_source_files"
    
    try:
        x_train, y_train, x_test, y_test = load_emnist_ubyte(data_folder)
        model = create_model()
        print("🚀 开始训练...")
        model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
        model.save('letter_ocr_model.h5')
        print("💾 字母模型已保存！")
    except Exception as e:
        print(f"❌ 运行出错：{e}")