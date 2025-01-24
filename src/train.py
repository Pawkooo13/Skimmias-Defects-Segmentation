import config as cfg
import os
import numpy as np


data_path = os.path.join(cfg.DATA_DIR, 'splitted_data.npy')

X_train, Y_train, X_test, Y_test = np.load(data_path, allow_pickle=True)

print(X_train.shape, X_test.shape)