import config as cfg
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def to_categorical(mask):
    mask_ = np.zeros((512, 512, 1), dtype=np.uint8)
    
    black = (mask == [0, 0, 0]).all(axis=-1)
    white = (mask == [255, 255, 255]).all(axis=-1)

    mask_[black] = 0  # tło
    mask_[white] = 1  # ugryzione
    mask_[~(black | white)] = 2  # spalone

    return mask_

def load_images_from_directory(directory, target_size):
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

    X_train, X_test, Y_train, Y_test = train_test_split(images_, 
                                                        categorical_masks, 
                                                        test_size=0.2, 
                                                        random_state=713)

    print("Wymiar zbioru treningowego:", (X_train.shape, Y_train.shape))
    print("Wymiar zbioru testowego:", (X_test.shape, Y_test.shape))
        
    np.save(f'{cfg.DATA_DIR}/splitted_data.npy', [X_train, Y_train, X_test, Y_test])


if __name__ == '__main__':
    main()
