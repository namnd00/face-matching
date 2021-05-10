import os
import pandas as pd
from tqdm import tqdm

for mset in ['train', 'test']:
    path = "../datasets/%s" % mset
    df_images = {"images": [], "quantity": []}

    for folder in tqdm(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        count = 0
        for ix, file in enumerate(os.listdir(folder_path)):
            new_file = "{}.{}.jpg".format(folder, ix + 1)
            file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(folder_path, new_file)
            os.rename(file_path, new_file_path)
            count += 1
        df_images["images"].append(folder)
        df_images["quantity"].append(count)
    df_mset = pd.DataFrame(data=df_images)
    df_name = "df_images_{}.csv".format(mset)
    df_mset.to_csv(os.path.join("../datasets", df_name))

