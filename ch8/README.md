# Ch8. Model of Keras Application
## Introduction
* Keras에서는 Imagenet으로 여러 CNN기반의 모델을 훈련시켜 가중치를 초기화한 모델을 제공한다.
* 대용량의 데이터로 사전학습을 해놓았기 때문에 더 빠르게 원하는 값에 수렴시킬 수 았다는 장점이 있다.
* 아래 사이트에 접속하면 다양한 모델을 확인할 수 있다.
[케라스 어플리케이션 페이지](https://keras.io/api/applications/)

## Load Model
VGG16 모델을 기준으로 진행해보겠다.
```python
import tensorflow as tf
model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            classes=1000,
            classifier_activation="softmax",
        )
```
다음과 같은 일련의 과정으로 모델을 로드할 수 있다.
* include_top = 마지막 classification layer를 넣을지 말지를 나타낸다. Fine-tuning을 위해 False로 설정한다.
* weights = 가중치로 무엇을 쓸건지를 의미한다. "imagenet"의 경우 imagenet가중치를 쓰는 것이고 안쓰면 None으로 세팅한다.
* input_tensor = 인풋으로 Tensor값을 넘겨줄 때 쓰는 파라미터이다.
* input_shape = 인풋 모양을 지정하여 서브 모델로 정의할 수 있다.
* classes = 몇개의 클래스를 분류하도록 할 것인지이다.
* classifier_activation = 무슨 activation을 마지막 레이어에 둘 것인지에 대한 세팅이다.

## Example of fine tunning
```python
import tensorflow as tf
model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=[224,224,3],
            classes=1000,
        ) # 가중치는 imagenet을 쓰지만 마지막 FC레이어를 제거해주고 로드한다.

x = model.output # 로드한 모델의 출력 레이어를 의미한다.

# 아래에 우리가 원하는 FC레이어를 붙이거나, vgg16을 feature extractor로 다른 레이어를 붙일 수 있도록 한다.
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5) 
out = tf.keras.layers.Dense(3, activation="softmax")(x)

new_model = tf.keras.Model(model.input, out)
```