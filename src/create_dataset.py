import numpy as np
import os
import random

from tqdm import tqdm


def prepare_data(path, haveVggface2=False):
    if haveVggface2:
        return

    _path = path
    _set = [f for f in os.listdir(_path) if not 'flip' in f and not "aug" in f]
    _male_set = [f for f in _set if "M" in f]
    _female_set = [f for f in _set if "F" in f]

    aug_set = [f for f in os.listdir(_path) if "aug" in f]
    aug_male_set = [f for f in aug_set if "M" in f]
    aug_female_set = [f for f in aug_set if "F" in f]

    assert len(aug_set) == len(_set)
    assert len(_male_set) == len(aug_male_set)
    assert len(_female_set) == len(aug_female_set)

    output_folder = path.split(os.path.sep)[-1]
    root_output_folder = path.split(os.path.sep)[-2]
    output_couples = f"../datasets/pre_data/{root_output_folder}/{output_folder}"
    output_diff = f"../datasets/pre_data/{root_output_folder}/{output_folder}"
    if not os.path.exists(output_couples):
        os.makedirs(output_couples)
    if not os.path.exists(output_diff):
        os.makedirs(output_diff)

    count_couples = 0

    print('[INFO] Preparing couples data...')
    with tqdm(total=len(_male_set)) as progressbar:
        for (ix, m_npy), m_aug_npy in zip(enumerate(_male_set), aug_male_set):
            for (ix2, f_npy), f_aug_npy in zip(enumerate(_female_set), aug_female_set):
                if m_npy[:11] == f_npy[:11]:
                    file = "{}.{:04}M.{:04}F.npy".format(m_npy[:11], ix + 1, ix2 + 1)
                    # file_2 = "{}.{:04}F.{:04}M.npy".format(m_npy[:11], ix2 + 1, ix + 1)
                    file_aug = "{}.{:04}M.{:04}F_aug.npy".format(m_aug_npy[:11], ix + 1, ix2 + 1)
                    # file_aug_2 = "{}.{:04}F.{:04}M_augmentation.npy".format(m_aug_npy[:11], ix2 + 1, ix + 1)

                    _male_path = os.path.join(_path, m_npy)
                    _female_path = os.path.join(_path, f_npy)
                    aug_male_path = os.path.join(_path, m_aug_npy)
                    aug_female_path = os.path.join(_path, f_aug_npy)

                    npy_of_male = np.load(_male_path)
                    npy_of_female = np.load(_female_path)
                    npy_aug_of_male = np.load(aug_male_path)
                    npy_aug_of_female = np.load(aug_female_path)

                    npy_concatenated = np.squeeze(np.hstack((npy_of_male, npy_of_female)))
                    # npy_concatenated_2 = np.squeeze(np.hstack((npy_of_female, npy_of_male)))
                    npy_aug_concatenated = np.squeeze(np.hstack((npy_aug_of_male, npy_aug_of_female)))
                    # npy_aug_concatenated_2 = np.squeeze(np.hstack((npy_aug_of_female, npy_aug_of_male)))

                    dims = npy_aug_concatenated.shape[-1]

                    norm_data = npy_concatenated / np.linalg.norm(npy_concatenated)
                    # norm_data_2 = npy_concatenated_2 / np.linalg.norm(npy_concatenated_2)
                    norm_aug_data = np.array([], dtype=np.float32).reshape(0, dims)
                    # norm_aug_data_2 = np.array([], dtype=np.float32).reshape(0, dims)

                    for i in range(100):
                        norm_aug_data = np.vstack((norm_aug_data,
                                                   npy_aug_concatenated[i] / np.linalg.norm(npy_aug_concatenated[i])))
                        # norm_aug_data_2 = np.vstack((norm_aug_data_2,
                        #                              npy_aug_concatenated_2[i] / np.linalg.norm(
                        #                                  npy_aug_concatenated_2[i])))

                    output_path = os.path.join(output_couples, file)
                    # output_path_2 = os.path.join(output_couples, file_2)
                    output_aug_path = os.path.join(output_couples, file_aug)
                    # output_aug_path_2 = os.path.join(output_couples, file_aug_2)

                    np.save(output_path, norm_data)
                    # np.save(output_path_2, norm_data_2)
                    np.save(output_aug_path, norm_aug_data)
                    # np.save(output_aug_path_2, norm_aug_data_2)

                    count_couples += 1
            progressbar.update(1)

    print('[INFO] Preparing non-couples data...')
    non_couples_list = []
    for i in tqdm(range(count_couples)):
        for j in range(100):
            x, y = random.sample(set(_set), 2)
            x_index, y_index = x.split(".")[1], y.split(".")[1]
            if 'M' in x and x[-8:] != y[-8:]:
                file = f"non.{x_index}M.{y_index}F.npy"
                # file_2 = f"non.{y_index}F.{x_index}M.npy"
                aug = f"non.{x_index}M.{y_index}F_aug.npy"
                # aug_2 = f"non.{y_index}F.{x_index}M_augmentation.npy"
                if not file in non_couples_list \
                        and not aug in non_couples_list:
                    non_couples_list.append(file)
                    # non_couples_list.append(file_2)
                    non_couples_list.append(aug)
                    # non_couples_list.append(aug_2)

                    aug_x = x.replace(".npy", "_aug.npy")
                    aug_y = y.replace(".npy", "_aug.npy")

                    male_path = os.path.join(_path, x)
                    female_path = os.path.join(_path, y)
                    male_aug_path = os.path.join(_path, aug_x)
                    female_aug_path = os.path.join(_path, aug_y)

                    npy_male = np.load(male_path)
                    npy_female = np.load(female_path)
                    npy_aug_male = np.load(male_aug_path)
                    npy_aug_female = np.load(female_aug_path)

                    npy_concatenated_diff = np.squeeze(np.hstack((npy_male, npy_female)))
                    # npy_concatenated_diff_2 = np.squeeze(np.hstack((npy_female, npy_male)))
                    npy_aug_concatenated_diff = np.squeeze(np.hstack((npy_aug_male, npy_aug_female)))
                    # npy_aug_concatenated_diff_2 = np.squeeze(np.hstack((npy_aug_female, npy_aug_male)))

                    dims = npy_aug_concatenated_diff.shape[-1]

                    norm_data_diff = npy_concatenated_diff / np.linalg.norm(npy_concatenated_diff)
                    # norm_data_diff_2 = npy_concatenated_diff_2 / np.linalg.norm(npy_concatenated_diff_2)
                    norm_aug_data_diff = np.array([], dtype=np.float32).reshape(0, dims)
                    # norm_aug_data_diff_2 = np.array([], dtype=np.float32).reshape(0, dims)

                    for ix in range(100):
                        norm_aug_data_diff = np.vstack((norm_aug_data_diff,
                                                        npy_aug_concatenated_diff[ix] / np.linalg.norm(
                                                            npy_aug_concatenated_diff[ix])))
                        # norm_aug_data_diff_2 = np.vstack((norm_aug_data_diff_2,
                        #                                   npy_aug_concatenated_diff_2[ix] / np.linalg.norm(
                        #                                       npy_aug_concatenated_diff_2[ix])))

                    output_diff_path = os.path.join(output_diff, file)
                    # output_diff_path_2 = os.path.join(output_diff, file_2)
                    output_diff_aug_path = os.path.join(output_diff, aug)
                    # output_diff_aug_path_2 = os.path.join(output_diff, aug_2)

                    np.save(output_diff_path, norm_data_diff)
                    # np.save(output_diff_path_2, norm_data_diff_2)
                    np.save(output_diff_aug_path, norm_aug_data_diff)
                    # np.save(output_diff_aug_path_2, norm_aug_data_diff_2)

                    break


def main():
    root_path = "../embeddings"

    for child_path in os.listdir(root_path)[4:]:
        # child_path = os.listdir(root_path)[0]
        print(child_path.split(os.path.sep)[-1])
        haveVggface2 = False
        if "vggface2" in child_path:
            haveVggface2 = True

        child_path = os.path.join(root_path, child_path)
        for sub_path in os.listdir(child_path):
            path = os.path.join(child_path, sub_path)
            # if os.path.exists(path) and len(os.listdir(path)) == 0:
            #     continue
            prepare_data(path, haveVggface2)


if __name__ == "__main__":
    main()
