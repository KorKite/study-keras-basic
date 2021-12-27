# 출처: https://www.tensorflow.org/tutorials/keras/classification?hl=ko
# 모델은 간단한 CNN레이어로 변경

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_model():
    inp = tf.keras.layers.Input([28,28])
    x = tf.keras.layers.Conv2D(8, (3,3), padding="same")(x)
    x = tf.keras.layers.Conv2D(8, (3,3), padding="same")(x)
    x = tf.keras.layers.Maxpooling2D(8, (3,3), padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(10, activation="softmax")

    return tf.keras.Model(inp, out)

model = get_model()
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["acc"]
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=100,
    validation_split=0.3,
    batch_size=16
)

y_pred = model.predict(test_images)
print(y_pred.shape)
print(test_labels.shape)

## 모델 평가
from sklearn.metrics import classification_report
print(classification_report(test_labels, y_pred))