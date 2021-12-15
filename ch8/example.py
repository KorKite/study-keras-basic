import tensorflow as tf
model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=[224,224,3],
        ) # 가중치는 imagenet을 쓰지만 마지막 FC레이어를 제거해주고 로드한다.

x = model.output # 로드한 모델의 출력 레이어를 의미한다.

# 아래에 우리가 원하는 FC레이어를 붙이거나, vgg16을 feature extractor로 다른 레이어를 붙일 수 있도록 한다.
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5) 
out = tf.keras.layers.Dense(3, activation="softmax")(x)

new_model = tf.keras.Model(model.input, out)
new_model.summary()