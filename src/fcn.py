from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, Add

class FCN:
    @staticmethod
    def conv_block(n_filters, x):
        x = Conv2D(n_filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(n_filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = MaxPool2D((2,2))(x)

        return x
    
    @staticmethod
    def encoder(inputs):
        c1 = FCN.conv_block(n_filters=16, x=inputs)
        c2 = FCN.conv_block(n_filters=32, x=c1)
        c3 = FCN.conv_block(n_filters=64, x=c2)
        c4 = FCN.conv_block(n_filters=128, x=c3)
        c5 = FCN.conv_block(n_filters=128, x=c4)

        c6 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c6)
        c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        return c6
    
    @staticmethod
    def t_conv_block(n_filters, x):
        u = Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same')(x)
        c = Conv2D(n_filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
        c = Add()([u, c])

        return c
    
    @staticmethod
    def decoder(encoder_output):
        c7 = FCN.t_conv_block(n_filters=64, x=encoder_output)
        c8 = FCN.t_conv_block(n_filters=32, x=c7)
        c9 = FCN.t_conv_block(n_filters=16, x=c8)
        c10 = FCN.t_conv_block(n_filters=16, x=c9)

        outputs = Conv2D(3, (1,1), activation='softmax')(c10)
        return outputs
    
    def build_model():
        inputs = Input(shape=(512,512,3))
        features = FCN.encoder(inputs)
        outputs = FCN.decoder(features)

        fcn_model = Model(inputs, outputs, name='FCN')
        return fcn_model