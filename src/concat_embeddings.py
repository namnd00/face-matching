import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch


def validate_embedding(embedding):
    if len(np.asarray(embedding).shape) < 3:
        return torch.from_numpy(np.transpose(np.asarray(embedding)[..., np.newaxis], (0, 2, 1)))
    else:
        return torch.from_numpy(np.asarray(embedding))


def get_embeddings_in_models(model_path, mset):
    pretrained_model_names = os.listdir(model_path)
    pretrained_model_paths = [os.path.join(model_path, p) for p in pretrained_model_names]
    pretrained_model_dir = [os.path.join(p, mset) for p in pretrained_model_paths]

    embedding_vectors_list = list()
    for pretrained_model in pretrained_model_dir:
        embedding_vectors = list()
        for r, _, files in os.walk(pretrained_model):
            for file in tqdm(files):
                if "augmentation" not in file and "flip" not in file:
                    file_path = os.path.join(r, file)
                    npy = np.load(file_path)
                    embedding_vectors.append(npy)
        embedding_vectors_list.append(embedding_vectors)

    assert len(embedding_vectors_list) == len(pretrained_model_names)
    return embedding_vectors_list, pretrained_model_names


def main():
    root_path = "../embeddings"
    output = "../embeddings_matched"
    csv = "../datasets/df_faces_train.csv"
    mset = "train"
    df = pd.read_csv(csv)
    files = [f.split('.jpg')[0] for f in df['faces']]
    model_paths = [os.path.join(root_path, p) for p in os.listdir(root_path)]

    embeddings_aggregated = list()
    pretrained_models_aggregated = list()
    for model_path in model_paths:
        embedding_vectors_list, pretrained_model_names = get_embeddings_in_models(model_path, mset)
        embeddings_aggregated.append(embedding_vectors_list)
        pretrained_models_aggregated.append(pretrained_model_names)

    print('[INFO] Concatenate face embeddings...')
    i = 1
    for embed1, name1 in zip(embeddings_aggregated[0], pretrained_models_aggregated[0]):
        for embed2, name2 in zip(embeddings_aggregated[1], pretrained_models_aggregated[1]):
            for embed3, name3 in zip(embeddings_aggregated[2], pretrained_models_aggregated[2]):
                for embed4, name4 in zip(embeddings_aggregated[3], pretrained_models_aggregated[3]):
                    for embed5, name5 in zip(embeddings_aggregated[4], pretrained_models_aggregated[4]):
                        for embed6, name6 in zip(embeddings_aggregated[5], pretrained_models_aggregated[5]):
                            for embed7, name7 in zip(embeddings_aggregated[6], pretrained_models_aggregated[6]):
                                for embed8, name8 in zip(embeddings_aggregated[7], pretrained_models_aggregated[7]):
                                    # names_list = [name1[:5], name2[:5], name3[:5], name4[:5], name5[:5], name6[:5], name7[:5], name8[:5]]
                                    match_name = f"ensemble_model_{i}"
                                    i += 1
                                    output_dir = os.path.join(output, match_name)
                                    if not os.path.exists(output_dir):
                                        os.makedirs(output_dir)

                                    embed1 = validate_embedding(embed1)
                                    embed2 = validate_embedding(embed2)
                                    embed3 = validate_embedding(embed3)
                                    embed4 = validate_embedding(embed4)
                                    embed5 = validate_embedding(embed5)
                                    embed6 = validate_embedding(embed6)
                                    embed7 = validate_embedding(embed7)
                                    embed8 = validate_embedding(embed8)

                                    match_embedding = [np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8), axis=1)
                                                       for (x1, x2, x3, x4, x5, x6, x7, x8) in zip(embed1,
                                                                                                   embed2,
                                                                                                   embed3,
                                                                                                   embed4,
                                                                                                   embed5,
                                                                                                   embed6,
                                                                                                   embed7,
                                                                                                   embed8)]
                                    assert len(files) == len(match_embedding)

                                    for emb, file in tqdm(zip(match_embedding, files), total=len(files)):
                                        np.save(os.path.join(output_dir, file), emb)


if __name__ == "__main__":
    main()