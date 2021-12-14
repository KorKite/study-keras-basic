import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3),padding="same"),
    tf.keras.layers.Conv2D(64, (3,3),padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3),padding="same"),
    tf.keras.layers.Conv2D(128, (3,3),padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3),padding="same"),
    tf.keras.layers.Conv2D(256, (3,3),padding="same"),
    tf.keras.layers.Conv2D(256, (3,3),padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.Conv2D(512, (3,3),padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dense(1000, activation="softmax")
])

model.build([None, 224,224,3])
model.summary()