# ultrasound-cnn-survey

This repository contains the code I used for my final project in the class Machine Learning for Health Care at NYU. You will find three subfolders:

## data

The `data` folder contains:
- a script written to convert the nerve data to the COCO format (`nerve_coco_data.py`)
- an `sbatch` file to run the script on NYU's HPC cluster (`coco.sbatch`)
- a script written to split the nerve data into train and test (`split_train_test.py`)
- an `sbatch` file to run the script on NYU's HPC cluster (`split.sbatch`)

Note that the `sbatch` files will need to be modified to contain the path to your nerve dataset and environments.

## unet

The `unet` folder contains:
- `train.py`: the python file containing the code to train the model
- `train.sbatch`: executes `train.py` on NYU's HPC cluster
- `batch_norm_train.py`: the python file containing the code to train the model with batchnorm
- `b-train.sbatch`: executes `batch_norm_train.py` on NYU's HPC cluster
- `data.py`: loads the data files into `.npy` format for faster loading
- `data.sbatch`: executes `data.py` on NYU's HPC cluster
- `get_masks.py`: loads in a trained model, computes the predicted masks on the test set, and saves the results
- `inspect_model.ipynb` allows you to interactively look at the results of your trained model
- `jupyter.sbatch` allows you to run jupyter notebooks on NYU's HPC cluster (requires extra config)

If you'd like to run training, you must download the dataset from the [Kaggle competition](https://www.kaggle.com/c/ultrasound-nerve-segmentation), run the `split_train_test.py` script, and either symlink or move `train` and `val` to `/path/to/repo/unet/raw`.

## mrcnn

This folder contains the implementation of the Mask R-CNN. This implementation was adapted from [Facebook's repository written in Pytorch 1.0](https://github.com/facebookresearch/maskrcnn-benchmark). I had to modify various internal files in order to get my code to work, so I copied their repository in here. Follow `INSTALL.md` to understand what needs to be downloaded in order to run training/evaluation.

This folder contains three relevant batch files:
- `train.sbatch`: trains a Mask R-CNN model on the nerve dataset
- `val.sbatch`: runs evaluation on a trained model
- `jupyter.sbatch`: allows you to run jupyter notebooks on NYU's HPC cluster (requires extra config)

The relevant jupyter notebook is found under `demos`: `Mask_R-CNN_demo.ipynb`

If you'd like to run training, you must download the dataset from the [Kaggle competition](https://www.kaggle.com/c/ultrasound-nerve-segmentation), run the `nerve_coco_data.py` script, and symlink `annotations`, `train`, and `val` to `/path/to/repo/mrcnn/datasets/nerve`.

