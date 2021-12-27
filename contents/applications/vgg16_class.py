import tensorflow as tf

class vgg(tf.keras.models.Model):
    def __init__(self):
        super(vgg,self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),padding="same"),
            tf.keras.layers.Conv2D(64, (3,3),padding="same"),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3,3),padding="same"),
            tf.keras.layers.Conv2D(128, (3,3),padding="same"),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.conv_block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, (3,3),padding="same"),
            tf.keras.layers.Conv2D(256, (3,3),padding="same"),
            tf.keras.layers.Conv2D(256, (3,3),padding="same"),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.conv_block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.conv_block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.Conv2D(512, (3,3),padding="same"),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.fc_layer = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(1000, activation="softmax")
        ])

    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.fc_layer(x)
        return x


model = vgg()
model.build((None,224,224,3))
model.summary()