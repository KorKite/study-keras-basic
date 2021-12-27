import tensorflow as tf
class autoencoder(tf.keras.models.Model):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(1024, 5, padding='same'),
            tf.keras.layers.Conv1D(512, 5, padding='same'),
            tf.keras.layers.Conv1D(256, 5, padding='same'),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(512, 5, padding='same'),
            tf.keras.layers.Conv1DTranspose(1024, 5, padding='same'),
            tf.keras.layers.Conv1DTranspose(5000, 5, padding='same'),
        ])
    
    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x): 
        # encoder함수를 새로 정의하여 훈련에는 encoder, decoder를 모두 활용해주지만,
        # 실제로 정보를 추출할 때에는 encoder만 활용해 줄 수 있도록 한다.
        return self.encoder(x)

ae = autoencoder()
ae.build((None, 12, 5000))
ae.summary()

# ae.encode(data)