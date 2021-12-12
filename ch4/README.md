# ch4 - Basic Keras Layers

## Convolution Based Layers
### 1. 1D Convolution Layer
    tf.keras.layers.Conv1D(
        filters, kernel_size, strides=1, padding='valid',
        data_format='channels_last', dilation_rate=1, groups=1,
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../figures/1d_convolution.png" width=500>


### 2. 2D Convolution Layer
    tf.keras.layers.Conv2D(
        filters, kernel_size, strides=(1, 1), padding='valid',
        data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../figures/2d_convolution.png" width=500>


### 3. 3D Convolution Layer
    tf.keras.layers.Conv3D(
        filters, kernel_size, strides=(1, 1, 1), padding='valid',
        data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../figures/3d_convolution.gif" width=500>
