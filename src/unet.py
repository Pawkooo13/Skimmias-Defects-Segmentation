from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, Concatenate

class UNet:
    @staticmethod
    def conv_block(n_filters, x):
        x = Conv2D(n_filters, (3,3), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x = Conv2D(n_filters, (3,3), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        return x
    
    @staticmethod
    def downsample_block(n_filters, x):
        f = UNet.conv_block(n_filters, x)
        p = MaxPool2D(2)(f)
        p = Dropout(0.2)(p)
        return f, p 
    
    @staticmethod
    def upsample_block(n_filters, conv_features, x):
        f = Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding="same")(x)
        f = Concatenate()([f, conv_features])
        f = UNet.conv_block(n_filters, f)
        f = Dropout(0.2)(f)
        return f

    def build_model():
        inputs = Input(shape=(512,512,3))
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = UNet.downsample_block(n_filters=32, x=inputs)
        # 2 - downsample
        f2, p2 = UNet.downsample_block(n_filters=32, x=p1)
        # 3 - downsample
        f3, p3 = UNet.downsample_block(n_filters=64, x=p2)
        # 4 - downsample
        f4, p4 = UNet.downsample_block(n_filters=64, x=p3)
        # 5 - downsample
        f5, p5 = UNet.downsample_block(n_filters=128, x=p4)
        # 6 - downsample
        f6, p6 = UNet.downsample_block(n_filters=128, x=p5)

        # bottleneck
        bottleneck = UNet.conv_block(n_filters=256, x=p6)

        # decoder: expanding path - upsample
        # 7 - upsample
        u7 = UNet.upsample_block(n_filters=128, conv_features=f6, x=bottleneck)
        # 8 - upsample
        u8 = UNet.upsample_block(n_filters=64, conv_features=f5, x=u7)
        # 9 - upsample
        u9 = UNet.upsample_block(n_filters=64, conv_features=f4, x=u8)
        # 10 - upsample
        u10 = UNet.upsample_block(n_filters=32, conv_features=f3, x=u9)
        # 11 - upsample
        u11 = UNet.upsample_block(n_filters=32, conv_features=f2, x=u10)
        # 12 - upsample
        u12 = UNet.upsample_block(n_filters=32, conv_features=f1, x=u11)
        # outputs
        outputs = Conv2D(3, 1, padding="same", activation = "softmax")(u12)
        
        unet_model = Model(inputs, outputs, name="U-Net")
        return unet_model