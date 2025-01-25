import config as cfg
import os
import numpy as np
from unet import UNet
from fcn import FCN

data_path = os.path.join(cfg.DATA_DIR, 'splitted_data.npy')

X_train, Y_train, X_test, Y_test = np.load(data_path, allow_pickle=True)

# uczenie modelu unet

print("Inicjalizowanie modelu UNet! \n")

unet_model = UNet.build_model()

unet_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

print("Uczenie modelu UNet: \n")

unet_history = unet_model.fit(x=X_train,
                              y=Y_train, 
                              batch_size=4,
                              epochs=100)

#uczenie modelu fcn

print("Inicjalizowanie modelu FCN! \n")

fcn_model = FCN.build_model()

fcn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("Uczenie modelu FCN: \n")

fcn_history = fcn_model.fit(x=X_train,
                            y=Y_train,
                            batch_size=4,
                            epochs=100)