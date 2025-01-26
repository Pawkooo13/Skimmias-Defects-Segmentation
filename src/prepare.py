import configs as cfg
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dvc.api import params_show

def to_categorical(mask):
    """
        mask - maska w formacie rgb

        Funkcja zwraca maske w formie kategorycznej (w,h,3) -> (w,h,1)
        np. [[0,0,0], ... , [0,0,0]] -> [[0], ... , [0]]
    """
    mask_ = np.zeros((512, 512, 1), dtype=np.uint8)
    
    black = (mask == [0, 0, 0]).all(axis=-1)
    white = (mask == [255, 255, 255]).all(axis=-1)

    mask_[black] = 0  # tło
    mask_[white] = 1  # ugryzione
    mask_[~(black | white)] = 2  # spalone

    return mask_

def load_images_from_directory(directory, target_size):
    """
        directory - folder, z którego wczytujemy dane
        target_size - wymiar wczytywanych zdjęć 

        Funkcja wczytuje wszystkie zdjęcia z podanego folderu i zwraca je w postaci listy
    """
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=target_size)  
        img_array = img_to_array(img, dtype=np.uint8)
        images.append(img_array)
    return np.array(images)


def main():

    images = load_images_from_directory(cfg.IMAGES_DIR, (512,512))
    masks = load_images_from_directory(cfg.MASKS_DIR, (512,512))

    images_ = np.array([image/255.0 for image in images])
    categorical_masks = np.array([to_categorical(mask) for mask in masks])

    params = params_show()
    test_size = params['split_data']['test_size']

    X_train, X_test, Y_train, Y_test = train_test_split(images_, 
                                                        categorical_masks, 
                                                        test_size=test_size, 
                                                        random_state=713)

    print("Wymiar zbioru treningowego:", (X_train.shape, Y_train.shape))
    print("Wymiar zbioru testowego:", (X_test.shape, Y_test.shape))
        
    np.save(f'{cfg.DATA_DIR}/splitted_data.npy', [X_train, Y_train, X_test, Y_test])

    ##################################################################################

    images_no_hard_mining = []
    masks_no_hard_mining = []

    for i in range(len(categorical_masks)):
        if (masks[i] == np.zeros(shape=(512,512,3))).all():
            continue
        else:
            images_no_hard_mining.append(images_[i])
            masks_no_hard_mining.append(categorical_masks[i])

    X_train_simplified, X_test_simplified, Y_train_simplified, Y_test_simplified = train_test_split(np.array(images_no_hard_mining), 
                                                                                                    np.array(masks_no_hard_mining), 
                                                                                                    test_size=test_size, 
                                                                                                    random_state=713)
    
    print("Wymiar zbioru treningowego bez zdjec hard-mining:", (X_train_simplified.shape, Y_train_simplified.shape))
    print("Wymiar zbioru testowego bez zdjec hard-mining:", (X_test_simplified.shape, Y_test_simplified.shape))

    np.save(f'{cfg.DATA_DIR}/splitted_no_hard_mining_data.npy', [X_train_simplified, Y_train_simplified, 
                                                                 X_test_simplified, Y_test_simplified])

if __name__ == '__main__':
    main()
