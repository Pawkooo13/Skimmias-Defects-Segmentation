import config as cfg
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def to_categorical(mask):
    mask_ = np.zeros((512, 512, 1), dtype=np.uint8)
    
    black = (mask == [0, 0, 0]).all(axis=-1)
    white = (mask == [255, 255, 255]).all(axis=-1)

    mask_[black] = 0  # tło
    mask_[white] = 1  # ugryzione
    mask_[~(black | white)] = 2  # spalone

    return mask_

def main():
    image_names = os.listdir(cfg.IMAGES_DIR)
    mask_names = os.listdir(cfg.MASKS_DIR)

    images = np.array([np.asarray(Image.open(os.path.join(cfg.IMAGES_DIR, image))) 
                    for image in image_names])
    masks = np.array([np.asarray(Image.open(os.path.join(cfg.MASKS_DIR, mask)))
                    for mask in mask_names])

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
