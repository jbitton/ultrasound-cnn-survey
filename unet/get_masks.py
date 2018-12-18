import numpy as np
from train import get_unet, preprocess

from data import load_test_data, load_train_data

imgs_train, imgs_mask_train = load_train_data()
imgs_train = preprocess(imgs_train)
imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)
std = np.std(imgs_train)

imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)
imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

model = get_unet()
model.load_weights("final.h5")
imgs_mask_test = model.predict(imgs_test, verbose=1)

np.save('imgs_mask_test_final.npy', imgs_mask_test)
