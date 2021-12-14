import tensorflow as tf

# Example 1
inp = tf.keras.layers.Input([128])
x = tf.keras.layers.Dense(128)(inp)
out = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.summary()




# Example2 - Multiple Input
inp1 = tf.keras.layers.Input([128]) # 1번 인풋
x1 = tf.keras.layers.Dense(32)(inp1) # Dense를 통과시킨 output

inp2 = tf.keras.layers.Input([32]) # 2번 인풋
x2 = tf.keras.layers.Dense(128)(inp2) # Dense를 통과시킨 output

x = tf.keras.layers.Concatenate(axis=1)([x1, x2]) # 두 레이어의 출력을 통합
out = tf.keras.layers.Dense(3, activation="softmax")(x) # 결과 출력층

model = tf.keras.Model([inp1, inp2], out)
model.summary()



# Example3 - Multiple Output
inp = tf.keras.layers.Input([128])
x = tf.keras.layers.Dense(128)(inp)
out1 = tf.keras.layers.Dense(3, activation="softmax")(x)
out2 = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, [out1,out2])
model.summary()