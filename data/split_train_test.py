import os
import argparse
import cv2


def validate_folders(data_folder, split_folder):
    if not os.path.exists(data_folder):
        raise (RuntimeError(f"Could not find {data_folder}, please download the data from Kaggle"))

    if not os.path.exists(split_folder):
        print(f"Could not find {split_folder}. Creating new folder and processing data")
        os.mkdir(split_folder)

    if not os.path.exists(os.path.join(split_folder, "train")):
        os.mkdir(os.path.join(split_folder, "train"))

    if not os.path.exists(os.path.join(split_folder, "val")):
        os.mkdir(os.path.join(split_folder, "val"))


def main():
    parser = argparse.ArgumentParser(description='Processing Nerve Data to Train/Test split')
    parser.add_argument('--data', type=str, default='nerve_data', metavar='d', help="downloaded data directory")
    parser.add_argument('--save', type=str, default='nerve_split_data', metavar='s', help="directory where data saved.")
    parser.add_argument('--split', type=int, default=10, metavar='S', help="# to split b/t train/val. def: 10")
    args = parser.parse_args()

    root_dir = os.getcwd()
    data_folder = os.path.join(root_dir, args.data)
    split_folder = os.path.join(root_dir, args.save)
    data_split = args.split

    validate_folders(data_folder, split_folder)

    train_folder = os.path.join(split_folder, "train")
    val_folder = os.path.join(split_folder, "val")

    for filename in os.listdir(data_folder):
        if '_mask.tif' not in filename and '.tif' in filename:
            case_number = int(filename.split('_')[0])
            image = cv2.imread(os.path.join(data_folder, filename))
            mask_file = filename.split('.')[0] + '_mask.tif'
            mask = cv2.imread(os.path.join(data_folder, mask_file))
            if case_number % data_split == 0:
                cv2.imwrite(os.path.join(val_folder, filename), image)
                cv2.imwrite(os.path.join(val_folder, mask_file), mask)
            else:
                cv2.imwrite(os.path.join(train_folder, filename), image)
                cv2.imwrite(os.path.join(train_folder, mask_file), mask)

if __name__ == "__main__":
    main()

