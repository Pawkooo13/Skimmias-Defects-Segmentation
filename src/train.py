import config as cfg
import os
import numpy as np
from unet import UNet
from fcn import FCN

def make_predictions(model, X_test):
    preds = []

    for img in X_test:
        pred = model.predict(img.reshape(1,512,512,3))
        class_mask = np.argmax(pred, axis=-1)
        preds.append(class_mask.reshape(512,512,1))

    return preds

def get_accuracy(y_true, y_pred):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(y_true, y_pred)
    return metric.result()

def main():

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
    
    unet_preds = make_predictions(model=unet_model, X_test=X_test)
    unet_accuracy = get_accuracy(y_true=Y_test, y_pred=unet_preds)

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
    
    fcn_preds = make_predictions(model=fcn_model, X_test=X_test)
    fcn_accuracy = get_accuracy(y_true=Y_test, y_pred=fcn_preds)
    
if __name__ == '__main__':
    main()