from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'raw/'

image_rows = 420
image_cols = 580


def create_data(folder):
    split_data_path = os.path.join(data_path, folder)
    images = os.listdir(split_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print(f"Creating {folder} images...")
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img_id = int(image_name.split('.')[0])
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(split_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(split_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_id[i] = img_id
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(f"imgs_{folder}.npy", imgs)
    np.save(f"imgs_id_{folder}.npy", imgs_id)
    np.save(f"imgs_mask_{folder}_gt.npy", imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train_gt.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_data("train")
    create_data("test")

