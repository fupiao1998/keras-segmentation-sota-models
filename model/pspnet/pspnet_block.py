from model.layers.conv_layers import Conv_bn_relu
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import concatenate


def pymraid_block(pool_size, pool_strides, stage):
    def layer(input_tensor):
        x = AveragePooling2D(pool_size=pool_size, strides=pool_strides)(input_tensor)
        x = Conv_bn_relu(num_filters=512, kernel_size=1, strides=1, stage=stage)(x)
        x = UpSampling2D(size=(pool_size, pool_size))(x)
        return x
    return layer


def pymraid():
    def layer(input_tensor):
        x_1 = pymraid_block(pool_size=60, pool_strides=60, stage='_py1_')(input_tensor)
        x_2 = pymraid_block(pool_size=30, pool_strides=30, stage='_py2_')(x_1)
        x_3 = pymraid_block(pool_size=20, pool_strides=20, stage='_py3_')(x_2)
        x_4 = pymraid_block(pool_size=10, pool_strides=10, stage='_py4_')(x_3)
        x_5 = Conv_bn_relu(512, 1, '_py5_')(x_4)
        x = concatenate([x_1, x_2, x_3, x_4, x_5])
        return x
    return layer