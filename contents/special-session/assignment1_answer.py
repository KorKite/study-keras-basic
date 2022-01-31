import tensorflow as tf

classes = 4

def maxpool_conv(layer, filters):
    mx = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(layer)
    conv = tf.keras.layers.Conv2D(filters, (3,3), padding="same", activation="relu")(mx)
    return conv

def block_conv1(layer, conv_filter):
    mx_conv = maxpool_conv(layer, conv_filter)
    conv = tf.keras.layers.Conv2D(conv_filter, (3,3), padding="same", activation="relu")(mx_conv)
    return conv

def block_conv2(layer, conv_filter):
    mx_conv = maxpool_conv(layer, conv_filter)
    conv = tf.keras.layers.Conv2D(conv_filter, (3,3), padding="same", activation="relu")(mx_conv)
    conv = tf.keras.layers.Conv2D(conv_filter, (3,3), padding="same", activation="relu")(conv)
    return conv

def block_deconv1(layer):
    upsam = tf.keras.layers.UpSampling2D(size=(2, 2))(layer)
    deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(upsam)
    deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(deconv)
    return deconv

def block_deconv2(layer):
    upsam = tf.keras.layers.UpSampling2D(size=(2, 2))(layer)
    deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(upsam)
    deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(deconv)
    deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(deconv)
    return deconv


inp = tf.keras.layers.Input([224, 224, 3])
conv = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(inp)
conv = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(conv)

conv = block_conv1(conv, 128)
conv1 = block_conv2(conv, 256)
conv2 = block_conv2(conv1, 512)
conv3 = block_conv2(conv2, 512)

z3 = maxpool_conv(conv1, classes)
z2 = maxpool_conv(conv2, classes)
z1 = maxpool_conv(conv3, 4096)

# Main Stream
ms = tf.keras.layers.Conv2D(4096, (3,3), padding="same", activation="relu")(z1)
ms = tf.keras.layers.Conv2D(4096, (3,3), padding="same", activation=None)(ms)

deconv1 = block_deconv2(ms)
add1 = z2 + deconv1
deconv2 = block_deconv2(add1)
add2 = z3 + deconv2

deconv = block_deconv2(add2)
deconv = block_deconv1(deconv)

deconv  = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv)
deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(deconv)
deconv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size=(3,3), padding="same")(deconv)


model = tf.keras.Model(inp, deconv)
model.summary()