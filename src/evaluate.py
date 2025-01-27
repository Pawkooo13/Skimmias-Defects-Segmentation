from configs import DATA_DIR, MODELS_DIR, RESULTS_DIR
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Accuracy

def make_predictions(model, data):
    """
        model - model, który będzie dokonywał predykcji
        data - dane, dla których model ma zwrócić predykcję 

        Funkcja zwraca predykcje w postaci maski przez podany model na wskazanym zbiorze danych
    """
    preds = []

    for img in data:
        pred = model.predict(img.reshape(1,512,512,3), verbose=0)
        class_mask = np.argmax(pred, axis=-1)
        preds.append(class_mask.reshape(512,512,1))

    return preds

def get_accuracy(y_true, y_pred):
    """
        y_true - zbiór wartości prawdziwych
        y_pred - zbiór wartości przewidywanych

        Funkcja zwraca dokładność pomiędzy zbiorem y_true a y_pred
    """
    metric = Accuracy()
    metric.update_state(y_true, y_pred)
    return metric.result().numpy()

def main():
    data_path = os.path.join(DATA_DIR, 'splitted_data.npy')
    data_no_hard_mining_path = os.path.join(DATA_DIR, 'splitted_no_hard_mining_data.npy') 
    
    X_test, Y_test = np.load(data_path, allow_pickle=True)[2:]
    X_test_smp, Y_test_smp = np.load(data_no_hard_mining_path, allow_pickle=True)[2:]

    models = os.listdir(MODELS_DIR)

    models_accuracy = {}
    for model_file in models:

        model_path = os.path.join(MODELS_DIR, model_file)
        print(model_path)
        model = load_model(model_path)

        if 'smp' in model_file:
            X, Y = (X_test_smp, Y_test_smp)
        else:
            X, Y = (X_test, Y_test)

        preds = make_predictions(model=model, data=X)
        acc = get_accuracy(y_true=Y, y_pred=preds)

        models_accuracy.update({model_file[:-6]: acc})

    results = pd.DataFrame.from_dict(models_accuracy, orient='index', columns=['accuracy'])
    print('Dokładność modeli na zbiorze testowym:')
    print(results.sort_values(by='accuracy', ascending=False))

    results_save_path = os.path.join(RESULTS_DIR, 'test_accuracies.csv')
    results.to_csv(results_save_path)

if __name__ == '__main__':
    main()
    
