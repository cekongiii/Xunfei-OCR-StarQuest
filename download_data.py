import tensorflow_datasets as tfds

def load_data():
    print("⏳ 正在加载轻量级 MNIST 数据集...")
    # 直接使用 tf.keras 自带的数据集，不需要额外下载巨大的 tfds
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 转换为 tf.data.Dataset 格式，方便后续处理
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return ds_train, ds_test, None

def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        # 因为 MNIST 只有数字 0-9，所以这里改成 10
        layers.Dense(10, activation='softmax') 
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model