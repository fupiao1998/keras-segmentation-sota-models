from model.layers.conv_layers import Conv_bn_relu
from model.layers.conv_layers import resnet_block
from model.layers.conv_layers import dense_block
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose


def conv_pool_block(num_filters, kernel_size, stage, batchnorm=False):
    def layer(inpurt_tensor):
        x = Conv_bn_relu(num_filters, kernel_size, batchnorm=batchnorm, stage=str(stage) + 'a')(inpurt_tensor)
        x = Conv_bn_relu(num_filters, kernel_size, batchnorm=batchnorm, stage=str(stage) + 'b')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    return layer


def resnet_pool_block(num_filters, kernel_size, stage):
    def layer(inpurt_tensor):
        x = resnet_block(num_filters, kernel_size, stage=str(stage) + 'a')(inpurt_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    return layer


def dense_pool_block(num_filters, kernel_size, stage):
    def layer(inpurt_tensor):
        x = dense_block(num_filters, kernel_size, stage=str(stage) + 'a')(inpurt_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    return layer


def Upsample2D_block(num_filters,
                     stage, kernel_size=(3,3),
                     upsample_rate=(2,2),
                     batchnorm=False, skip=None):
    def layer(input_tensor):
        x = UpSampling2D(size=upsample_rate, name='up_name'+str(stage))(input_tensor)
        if skip is not None:
            x = Concatenate()([x, skip])
        x = Conv_bn_relu(num_filters, kernel_size, stage=str(stage) + 'c', batchnorm=batchnorm)(x)
        x = Conv_bn_relu(num_filters, kernel_size, stage=str(stage) + 'd', batchnorm=batchnorm,)(x)
        return x
    return layer


def Transpose2D_block(num_filters,
                     stage, kernel_size=(3,3),
                     upsample_rate=(2,2),
                     transpose_kernel_size=(4,4),
                     batchnorm=False, skip=None):
    def layer(input_tensor):
        x = Conv2DTranspose(num_filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name='transpose'+str(stage), use_bias=not (batchnorm))(input_tensor)
        if batchnorm:
            x = BatchNormalization(name='bn' + str(stage))(x)
        x = Activation('relu')(x)
        if skip is not None:
            x = Concatenate()([x, skip])
        x = Conv_bn_relu(num_filters, kernel_size, stage=str(stage), batchnorm=batchnorm)(x)
        return x
    return layer