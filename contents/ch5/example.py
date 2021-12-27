import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(3, activation="softmax") # Linear Layer 2개를 임의로 쌓아줌
])
model.build([None, 128])
model.summary()