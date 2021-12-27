import tensorflow as tf
def vgg16(input_shape):
    inp = tf.keras.layers.Input(input_shape)
    
    conv1 = tf.keras.layers.Conv2D(64, (3,3),padding="same")(inp)
    conv1 = tf.keras.layers.Conv2D(64, (3,3),padding="same")(conv1)
    conv1 = tf.keras.layers.MaxPooling2D()(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (3,3),padding="same")(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3),padding="same")(conv2)
    conv2 = tf.keras.layers.MaxPooling2D()(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (3,3),padding="same")(conv2)
    conv3 = tf.keras.layers.Conv2D(256, (3,3),padding="same")(conv3)
    conv3 = tf.keras.layers.Conv2D(256, (3,3),padding="same")(conv3)
    conv3 = tf.keras.layers.MaxPooling2D()(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv3)
    conv4 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv4)
    conv4 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv4)
    conv4 = tf.keras.layers.MaxPooling2D()(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv4)
    conv5 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv5)
    conv5 = tf.keras.layers.Conv2D(512, (3,3),padding="same")(conv5)
    conv5 = tf.keras.layers.MaxPooling2D()(conv5)
    
    fc = tf.keras.layers.Flatten()(conv5)
    fc = tf.keras.layers.Dense(4096, activation="relu")(fc)
    fc = tf.keras.layers.Dense(4096, activation="relu")(fc)
    fc = tf.keras.layers.Dense(1000, activation="softmax")(fc)

    return tf.keras.Model(inp, fc)

vgg16([224,224,3]).summary()