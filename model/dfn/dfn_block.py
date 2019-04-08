from model.layers.conv_layers import Conv_bn_relu
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import multiply
from keras.layers import AveragePooling2D
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model


def RRB(num_filters, stage):
    def layer(input_tnesor):
        x = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input_tnesor)
        x1 = Conv_bn_relu(num_filters, kernel_size=3, stage=stage)(x)
        sum = add([x, x1])
        x = Activation('relu')(sum)
        return x
    return layer


def CAB(num_filters):
    def layer(right, bottom):
        x = concatenate([right, bottom], axis=3)   # shape of bottom should be half of right
        print(x.shape)
        x = AveragePooling2D(pool_size=(1, 1))(x)
        print(x.shape)
        x = Conv2D(num_filters, kernel_size=(1, 1), padding='valid', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=(1, 1), padding='valid', activation='sigmoid')(x)
        right = multiply([right, x])
        print(right.shape)
        out = concatenate([bottom, right], axis=3)
        return out
    return layer


def model():
    input1 = Input((256, 256, 128))
    input2 = Input((256, 256, 128))
    out = CAB(128)(input1, input2)
    model = Model([input1, input2], out)
    model.summary()

model()