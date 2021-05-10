import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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


def main():
    root_path = "../datasets/pre_data"
    output = "../datasets"
    _type = ".npy"

    # random_folder = random.choice(os.listdir(root_path))
    random_folder = os.listdir(root_path)[0]
    _path = random.choice(os.listdir(os.path.join(root_path, random_folder)))
    used_path = os.path.join(root_path, random_folder)

    sub_path = random.choice(os.listdir(used_path))
    name = sub_path.split(os.path.sep)[-1]
    print(name)

    print("[INFO] Loading data...")

    sub_path = os.path.join(used_path, sub_path)

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

    output_train_path = os.path.join(output, "train.csv")
    output_test_path = os.path.join(output, "sample_submission.csv")

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)


if __name__ == "__main__":
    main()
    print('[INFO] Completed.')
