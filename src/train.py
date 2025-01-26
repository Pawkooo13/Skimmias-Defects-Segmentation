from configs import DATA_DIR, PLOTS_DIR, MODELS_DIR
import os
import numpy as np
from unet import UNet
from fcn import FCN
import pandas as pd
from tensorflow.keras.metrics import Accuracy
from tensorflow.config import list_physical_devices

def make_predictions(model, X_test):
    preds = []

    for img in X_test:
        pred = model.predict(img.reshape(1,512,512,3))
        class_mask = np.argmax(pred, axis=-1)
        preds.append(class_mask.reshape(512,512,1))

    return preds

def get_accuracy(y_true, y_pred):
    metric = Accuracy()
    metric.update_state(y_true, y_pred)
    return metric.result()

def get_training_plot(history, filename):
    ax = pd.DataFrame(history.history).plot()
    fig = ax.get_figure()
    path = os.path.join(PLOTS_DIR, f'{filename}.png')
    fig.savefig(path)

def main():

    data_path = os.path.join(DATA_DIR, 'splitted_data.npy')
    data_no_hard_mining_path = os.path.join(DATA_DIR, 'splitted_no_hard_mining_data.npy')

    X_train, Y_train = np.load(data_path, allow_pickle=True)[:2]
    X_train_smp, Y_train_smp = np.load(data_no_hard_mining_path, allow_pickle=True)[:2]

    models = {'UNet': 0, 'FCN': 0, 'UNet_smp': 0, 'FCN_smp': 0}

    print("Liczba dostępnych GPU: ", len(list_physical_devices('GPU')))

    for model_name in models.keys():
        if 'smp' in model_name:
            X, Y = (X_train_smp, Y_train_smp)
        else:
            X, Y = (X_train, Y_train)

        print(f'Inicjalizowanie modelu {model_name}! \n')
        if 'FCN' in model_name:
            model = FCN.build_model()
        else:
            model = UNet.build_model()
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        print("Uczenie modelu: \n")

        history = model.fit(x=X,
                            y=Y, 
                            batch_size=4,
                            epochs=30)

        model_save_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
        model.save(model_save_path)

        get_training_plot(history=history, filename=model_name)

        acc = history.history['accuracy'][-1]
        models.update({model_name: acc})

    accuracies = pd.DataFrame.from_dict(models, orient='index', columns=['accuracy'])
    print(accuracies.sort_values(by='accuracy', ascending=False))

if __name__ == '__main__':
    main()