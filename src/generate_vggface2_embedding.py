from __future__ import absolute_import
from __future__ import print_function

import os
import PIL
import imp

import cv2
import torch
import glob as gb
import numpy as np
import torchvision

from PIL import Image
from tqdm import tqdm

# hyper parameters
batch_size = 10
mean = (131.0912, 103.8827, 91.4953)
to_tensor = torchvision.transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def rotate_channels(img):
    return PIL.Image.merge("RGB", (list(img.split()))[::-1])


def load_data(path='', shape=None):
    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)  # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),  # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=PIL.Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    h_start = (newshape[0] - crop_size[0]) // 2
    w_start = (newshape[1] - crop_size[1]) // 2
    x = x[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    x = x - mean
    return x


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def initialize_model(pretrained_model):
    # Download the pytorch model and weights.
    # Currently, it's cpu mode.
    MainModel = imp.load_source('MainModel',
                                '../models/vggface2/pretrained_models/%s/%s.py' % (pretrained_model, pretrained_model))

    if pretrained_model == 'resnet50_128':
        feat_dim = 128
        network = MainModel.resnet50_128(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'resnet50_256':
        feat_dim = 256
        network = MainModel.resnet50_256(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'resnet50_ft':
        feat_dim = 2048
        network = MainModel.resnet50_ft(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'resnet50_scratch':
        feat_dim = 2048
        network = MainModel.resnet50_scratch(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'senet50_128':
        feat_dim = 128
        network = MainModel.senet50_128(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'senet50_256':
        feat_dim = 256
        network = MainModel.senet50_256(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    elif pretrained_model == 'senet50_ft':
        feat_dim = 2048
        network = MainModel.senet50_ft(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))
    else:
        feat_dim = 2048
        network = MainModel.senet50_scratch(
            weights_path='../models/vggface2/pretrained_models/%s/%s.pth' % (pretrained_model, pretrained_model))

    network.eval()

    return network, feat_dim


def my_process_flip(img_path):
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = to_tensor(rotate_channels(img)) * 255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3, 1, 1))
    return img.numpy()


def image_encoding(model, facepaths, feat_dims, output_dir):
    num_faces = len(facepaths)
    face_feats = np.empty((num_faces, feat_dims))
    face_feats_flip = np.empty((num_faces, feat_dims))
    imgpaths = facepaths
    imgchunks = list(chunks(imgpaths, batch_size))

    with tqdm(total=len(facepaths)) as progressbar:
        progressbar.set_description("[INFO] Extracting")
        for c, imgs in enumerate(imgchunks):
            im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])
            im_flip = np.array([my_process_flip(img_file) for img_file in imgs])
            f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)))[1].detach().cpu().numpy()[:, :, 0, 0]
            f_flip = model(torch.Tensor(im_flip.transpose(0, 1, 3, 2)))[1].detach().cpu().numpy()[:, :, 0, 0]
            start = c * batch_size
            end = min((c + 1) * batch_size, num_faces)
            # This is different from the Keras model where the normalization has been done inside the model.
            face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
            face_feats_flip[start:end] = f / np.sqrt(np.sum(f_flip ** 2, -1, keepdims=True))
            progressbar.update(batch_size)

    for ix, (face_feat, facepath) in enumerate(zip(face_feats, facepaths)):
        np.save(output_dir + '/' + os.path.split(facepath)[-1].replace('.jpg', '.npy'), face_feat)
    for ix, (face_feat, facepath) in enumerate(zip(face_feats_flip, facepaths)):
        np.save(output_dir + '/' + os.path.split(facepath)[-1].replace('.jpg', '_flip.npy'), face_feat)


def main():
    pretrained_models_path = '../models/vggface2/pretrained_models'
    pretrained_models = os.listdir(pretrained_models_path)

    for mset in ['train', 'test']:
        faces = gb.glob(f'../datasets/aligned/224x224/{mset}/*.jpg')

        for pretrained_model in pretrained_models:
            print(pretrained_model)
            embedding_path = "../embeddings/vggface2/%s/%s" % (pretrained_model, mset)
            if not os.path.exists(embedding_path):
                os.makedirs(embedding_path)

            model_eval, feat_dims = initialize_model(pretrained_model)
            image_encoding(model_eval, faces, feat_dims, embedding_path)


if __name__ == '__main__':
    main()
