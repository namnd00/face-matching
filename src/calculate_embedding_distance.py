import numpy as np
import os
import argparse
import pandas as pd

from numpy.linalg import norm
from tqdm import tqdm
from glob import glob


def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def cosine(u, v):
    """
    Computes the Cosine distance between 1-D arrays.
    The Cosine distance between `u` and `v`, is defined as
    .. math::
       1 - \\frac{u \\cdot v}
                {||u||_2 ||v||_2}.
    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


def braycurtis(u, v):
    """
    Computes the Bray-Curtis distance between two 1-D arrays.
    Bray-Curtis distance is defined as
    .. math::
       \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}
    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    braycurtis : double
        The Bray-Curtis distance between 1-D arrays `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    dist = np.sum(abs(u - v)) / np.sum(abs(u + v))
    return dist


def canberra(u, v):
    """
    Computes the Canberra distance between two 1-D arrays.
    The Canberra distance is defined as
    .. math::
         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    canberra : double
        The Canberra distance between vectors `u` and `v`.
    Notes
    -----
    When `u[i]` and `v[i]` are 0 for given i, then the fraction 0/0 = 0 is
    used in the calculation.
    """
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    d = np.sum(abs(u - v) / (abs(u) + abs(v)))
    return d


def chebyshev(u, v):
    """
    Computes the Chebyshev distance.
    Computes the Chebyshev distance between two 1-D arrays `u` and `v`,
    which is defined as
    .. math::
       \\max_i {|u_i-v_i|}.
    Parameters
    ----------
    u : (N,) array_like
        Input vector.
    v : (N,) array_like
        Input vector.
    Returns
    -------
    chebyshev : double
        The Chebyshev distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = np.max(abs(u - v))
    return dist


def cityblock(u, v):
    """
    Computes the City Block (Manhattan) distance.
    Computes the Manhattan distance between two 1-D arrays `u` and `v`,
    which is defined as
    .. math::
       \\sum_i {\\left| u_i - v_i \\right|}.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    cityblock : double
        The City Block (Manhattan) distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = np.sum(abs(u - v))
    return dist


def correlation(u, v):
    """
    Computes the correlation distance between two 1-D arrays.
    The correlation distance between `u` and `v`, is
    defined as
    .. math::
       1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
               {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}
    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    umu = np.mean(u)
    vmu = np.mean(v)
    um = u - umu
    vm = v - vmu
    dist = 1.0 - np.dot(um, vm) / (norm(um) * norm(vm))
    return dist


def euclidean(u, v):
    """
    Computes the Euclidean distance between two 1-D arrays.
    The Euclidean distance between 1-D arrays `u` and `v`, is defined as
    .. math::
       {||u-v||}_2
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    euclidean : double
        The Euclidean distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = norm(u - v)
    return dist


def sqeuclidean(u, v):
    """
    Computes the squared Euclidean distance between two 1-D arrays.
    The squared Euclidean distance between `u` and `v` is defined as
    .. math::
       {||u-v||}_2^2.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    sqeuclidean : double
        The squared Euclidean distance between vectors `u` and `v`.
    """
    # Preserve float dtypes, but convert everything else to np.float64
    # for stability.
    utype, vtype = None, None
    if not (hasattr(u, "dtype") and np.issubdtype(u.dtype, np.inexact)):
        utype = np.float64
    if not (hasattr(v, "dtype") and np.issubdtype(v.dtype, np.inexact)):
        vtype = np.float64

    u = _validate_vector(u, dtype=utype)
    v = _validate_vector(v, dtype=vtype)
    u_v = u - v
    dist = np.dot(u_v, u_v)
    return dist


def get_embedding_distances(embedding_vectors):
    num_vectors = len(embedding_vectors)
    num_distances = np.ndarray((num_vectors, num_vectors, 8), dtype=np.float32)
    for i in tqdm(range(num_vectors)):
        temp_distances = list()
        for j in range(num_vectors):
            f1 = embedding_vectors[i]
            f2 = embedding_vectors[j]
            cosine_value = cosine(f1, f2)
            braycurtis_value = braycurtis(f1, f2)
            canberra_value = canberra(f1, f2)
            chebyshev_value = chebyshev(f1, f2)
            cityblock_value = cityblock(f1, f2)
            correlation_value = correlation(f1, f2)
            euclidean_value = euclidean(f1, f2)
            sqeuclidean_value = sqeuclidean(f1, f2)

            result = [cosine_value,
                      braycurtis_value,
                      canberra_value,
                      chebyshev_value,
                      cityblock_value,
                      correlation_value,
                      euclidean_value,
                      sqeuclidean_value]
            temp_distances.append(result)
        num_distances[i] = np.asarray(temp_distances)
    num_distances = np.transpose(num_distances, (2, 0, 1))
    return num_distances


def get_embedding_distances_results(embedding_distances, image_names, output, model):
    # forms = ['cosine']
    forms = ['cosine',
             'braycurtis',
             'canberra',
             'chebyshev',
             'cityblock',
             'correlation',
             'euclidean',
             'sqeuclidean']

    assert len(embedding_distances) == len(forms)

    with tqdm(total=len(embedding_distances)) as progressbar1:
        progressbar1.set_description("[INFO] Calculating")
        for embedding_distance_form, form in zip(embedding_distances, forms):
            assert len(embedding_distance_form) == len(image_names)

            couple_index_list = []
            couple_values_list = []

            diff_index_list = []
            diff_values_list = []

            with tqdm(total=len(embedding_distance_form)) as progressbar2:
                progressbar2.set_description("[INFO] Analyzing")
                for ix in range(len(image_names)):
                    for ix2 in range(len(image_names)):
                        if image_names[ix][:11] == image_names[ix2][:11]:
                            if image_names[ix][-5:] != image_names[ix2][-5:]:
                                key = image_names[ix] + "-" + image_names[ix2]
                                value = embedding_distance_form[ix][ix2]
                                couple_index_list.append(key)
                                couple_values_list.append(value)

                        else:
                            key = image_names[ix] + "-" + image_names[ix2]
                            value = embedding_distance_form[ix][ix2]
                            diff_index_list.append(key)
                            diff_values_list.append(value)

                    progressbar2.update(1)

            output_ = os.path.join(output, form)
            if not os.path.exists(output_):
                os.makedirs(output_)
            couples_output = os.path.join(output_, f"{model}_couples.csv")
            diff_output = os.path.join(output_, f"{model}_diff.csv")

            assert len(couple_index_list) == len(couple_values_list)
            assert len(diff_index_list) == len(diff_values_list)

            df_couples = pd.DataFrame(data=couple_values_list, index=couple_index_list)
            df_diff = pd.DataFrame(data=diff_values_list, index=diff_index_list)

            # perc = [.20, .40, .60, .80]
            df_couples.describe().to_csv(couples_output)
            df_diff.describe().to_csv(diff_output)

            progressbar1.update(1)


def parse_argument():
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--path', default='', help='path to load embedding files')
    return parser.parse_args()


def main():
    for root_path, i in zip(["../embeddings/*/*/train", "../embeddings_matched/*"], [3, 2]):
        paths = glob(root_path)
        for ix, path in enumerate(paths):
            path = os.path.normpath(path)
            # ../embeddings/{model}/..
            model = path.split(os.sep)[2]
            print(model)
            # pretrained_model = path.split(os.sep)[3]
            pretrained_model = path.split(os.sep)[i]
            output = f"../embedding_distance/{model}"
            if i == 2:
                pretrained_model = f"ensemble_model_{ix + 1}"
                model = pretrained_model
                output = f"../ensemble_model/{model}"
            image_names = list()
            for f in os.listdir(path):
                if "augmentation" not in f and "flip" not in f and ".npy" in f:
                    image_names.append(f)
            if not os.path.exists(output):
                os.makedirs(output)

            embedding_vectors = [np.load(os.path.join(path, file)) for file in image_names]
            embedding_distances = get_embedding_distances(embedding_vectors)
            get_embedding_distances_results(embedding_distances, image_names, output, pretrained_model)


if __name__ == "__main__":
    main()
