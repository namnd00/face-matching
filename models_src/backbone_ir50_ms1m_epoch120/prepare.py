import pandas as pd
import numpy as np
import os
import shutil

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import *
from multiprocessing import Pool, cpu_count
import random


# IMPORT_DIRECTORY = os.getcwd()
# data_dir = os.path.join(IMPORT_DIRECTORY, "data")
DIMS = 512 * 2
NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 10


def list_files(base_path, valid_exts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) != -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if valid_exts is None or ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path


def set_dataset(paths):
    random.shuffle(paths)
    set_names = [p.split(os.path.sep)[-1] for p in paths]
    set_labels = [(p.split(os.path.sep)[-1]).split('.')[0] for p in paths]

    rating = int(round(0.7 * len(set_names)))
    train, label_train = list(set_names[:rating]), list(set_labels[:rating])
    test, label_test = list(set_names[rating:]), list(set_labels[rating:])

    return train, label_train, test, label_test


def generate_data():
    root_path = "../datasets/pre_data"
    output = "../datasets"
    _type = ".npy"

    # random_folder = random.choice(os.listdir(root_path))
    sub_path = "../../datasets/pre_data/face.evoLVe.PyTorch/backbone_ir50_ms1m_epoch120"

    print("[INFO] Loading data...")

    npy_paths = [f for f in list(list_files(sub_path, _type)) if not "aug" in f]

    couple_path = [f_couple for f_couple in npy_paths if "couple" in f_couple.split(os.sep)[-1]]
    non_couple_path = [f_non_couple for f_non_couple in npy_paths if "non" in f_non_couple]

    dims = np.load(random.choice(npy_paths)).shape[-1]
    print("[INFO] Dimension: " + str(dims))

    train_couple, label_train_couple, test_couple, label_test_couple = set_dataset(couple_path)
    train_non_couple, label_train_non_couple, test_non_couple, label_test_non_couple = set_dataset(non_couple_path)

    print(type(train_couple))

    assert len(train_couple) == len(train_non_couple)
    assert len(label_train_couple) == len(label_train_non_couple)
    assert len(test_couple) == len(test_non_couple)
    assert len(label_test_couple) == len(label_test_non_couple)

    train = train_couple + train_non_couple
    test = test_couple + test_non_couple
    label_train = label_train_couple + label_train_non_couple
    label_test = label_test_couple + label_test_non_couple

    le = LabelEncoder()
    label_train = le.fit_transform(label_train)
    label_test = le.fit_transform(label_test)
    print(le.classes_)

    assert len(train) == len(label_train)
    assert len(test) == len(label_test)

    print(len(train), len(test))

    train_set_dict = {"name": train, "label": label_train}
    test_set_dict = {"name": test, "label": label_test}

    train_df = pd.DataFrame(data=train_set_dict)
    test_df = pd.DataFrame(data=test_set_dict)

    train_df.head()
    print("=========================================")
    test_df.head()

    output_train_path = "train.csv"
    output_test_path = "sample_submission.csv"

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)


def split_dataset():
    test_df = pd.read_csv('sample_submission.csv', usecols=['name'])
    test_df.to_csv('test_refined.csv', index=False)
    print(test_df.head(5))

    train_df = pd.read_csv('train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    indexs = list(range(train_df.shape[0]))
    for i in range(NUMBER_OF_PARTS):
        info = {}
        rt = random.randint(1, 99)
        kf = KFold(n_splits=NUMBER_OF_FOLDS, random_state=rt, shuffle=True)
        for fold, (_, valid_index) in enumerate(kf.split(indexs)):
            for vi in valid_index:
                info[vi] = fold
        myarr = []
        for idx in range(train_df.shape[0]):
            myarr.append(info[idx])
        train_df['rt%d' % i] = np.array(myarr)
    train_df.to_csv('train_refined.csv', index=False)
    print(train_df.head(5))


def my_process(file_name):
    emb_path = f"../../datasets/pre_data/face.evoLVe.PyTorch/backbone_ir50_ms1m_epoch120/{file_name}"
    # emb_path = os.path.join(data_dir, file_name)
    emb = np.load(emb_path).reshape(DIMS)
    return emb


def my_process_train_augmentation(file_name):
    emb_path = f"../../datasets/pre_data/face.evoLVe.PyTorch/backbone_ir50_ms1m_epoch120/{file_name.replace('.npy', '_aug.npy')}"
    # emb_path = os.path.join(data_dir, file_name.replace('.npy', '_aug.npy'))
    emb = np.load(emb_path).reshape(100, DIMS)
    return emb


def prepare_dataset():
    test_df = pd.read_csv('test_refined.csv')
    train_df = pd.read_csv('train_refined.csv')

    print("Read file successfully.")

    p = Pool(16)
    test_data = p.map(func=my_process, iterable=test_df.name.values.tolist())
    p.close()
    test_data = np.array(test_data)
    print(test_data.shape)
    np.save('test_data.npy', test_data)
    test_data = []
    print("[INFO] Completed test data.")

    p = Pool(16)
    train_data = p.map(func=my_process, iterable=train_df.name.values.tolist())
    p.close()
    train_data = np.array(train_data)
    print(train_data.shape)
    np.save('train_data.npy', train_data)
    train_data = []
    print("[INFO] Completed train data.")

    p = Pool(16)
    train_aug_data = p.map(func=my_process_train_augmentation, iterable=train_df.name.values.tolist())
    p.close()
    train_aug_data = np.array(train_aug_data)
    print(train_aug_data.shape)
    np.save('train_aug_data.npy', train_aug_data)
    train_aug_data = []
    print("[INFO] Completed train augmentation data.")

    shutil.rmtree(f"../../datasets/pre_data/face.evoLVe.PyTorch/backbone_ir50_ms1m_epoch120")


if __name__ == '__main__':
    generate_data()
    split_dataset()
    prepare_dataset()
