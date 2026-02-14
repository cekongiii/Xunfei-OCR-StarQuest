import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 加载数据 (改成了轻量级 MNIST)
def load_data():
    print("⏳ 正在加载轻量级 MNIST 数据集 (无需下载)...")
    # Keras 自带数据，只有 11MB，瞬间完成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 转换为 tf.data.Dataset 对象，方便后续流水线处理
    # 此时 x_train 形状是 (60000, 28, 28)，还没有通道维度
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_ds, test_ds

# 2. 定义模型 (输出改为 10 类)
def create_model():
    model = models.Sequential([
        # 显式定义输入层 (28宽, 28高, 1通道)
        layers.Input(shape=(28, 28, 1)),
        
        # 卷积层提取特征
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 展平并全连接
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        # 【关键修改】MNIST 只有数字 0-9，所以输出层是 10 个节点
        layers.Dense(10, activation='softmax') 
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. 训练逻辑 (增加了 expand_dims 处理维度)
def train_model(model, train_ds, test_ds):
    # 预处理函数
    def preprocess(image, label):
        # MNIST 数据原始形状是 (28, 28)，CNN 需要 (28, 28, 1)
        # 所以必须增加一个维度 (expand_dims)
        image = tf.expand_dims(image, axis=-1)
        # 归一化到 0-1 之间
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # 构建数据流水线：打乱 -> 分批 -> 预取
    # buffer_size=10000 表示打乱的程度
    train_ds = train_ds.map(preprocess).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    print("🚀 开始训练数字识别模型 (共 5 轮)...")
    # 开始训练
    model.fit(train_ds, epochs=5, validation_data=test_ds)
    
    # 保存模型
    model.save('my_ocr_model.h5')
    print("💾 模型已保存为 my_ocr_model.h5")

# 4. 主程序
if __name__ == "__main__":
    train_ds, test_ds = load_data()
    model = create_model()
    train_model(model, train_ds, test_ds)