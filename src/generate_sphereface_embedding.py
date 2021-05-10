import imp
import os

import torch
import numpy as np
import cv2

from torch.autograd import Variable
from glob import glob
from tqdm import tqdm
from imgaug import augmenters as iaa

batch_size = 10
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    sometimes(
        iaa.OneOf([
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.GaussianBlur((0, 2.0)),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
            iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
        ])
    )
])


def initialize_model():
    net_sphere = imp.load_source("sphere20a", "../models/sphereface_pytorch/net_sphere.py")
    net = net_sphere.sphere20a(feature=True)
    state_dict = torch.load("../models/sphereface_pytorch/pretrained_models/sphere20a_20171020/sphere20a_20171020.pth")
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()

    return net


def get_embedding(model, paths, output_dir):
    with tqdm(total=len(paths)) as progressbar:
        progressbar.set_description("[INFO] Extracting")
        for path in paths:
            image = cv2.imread(path)
            im_array = image.transpose(2, 0, 1).reshape((1, 3, 112, 96))
            with torch.no_grad():
                img = Variable(torch.from_numpy(im_array).float()).cuda()
            f = model(img).data.cpu().numpy()
            feature = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))

            augmentation_arr = np.array([], dtype=np.float32).reshape(0, 512)
            for j in range(100):
                img_aug = seq.augment_image(image)
                img_aug = img_aug.transpose(2, 0, 1).reshape((1, 3, 112, 96))
                with torch.no_grad():
                    img_aug = Variable(torch.from_numpy(img_aug).float()).cuda()
                f_aug = model(img_aug).data.cpu().numpy()
                feature_aug = f_aug / np.sqrt(np.sum(f_aug ** 2, -1, keepdims=True))
                augmentation_arr = np.vstack((augmentation_arr, feature_aug.reshape(1, 512)))

            np.save(output_dir + '/' + os.path.split(path)[-1].replace('.jpg', '.npy'), feature)
            np.save(output_dir + '/' + os.path.split(path)[-1].replace('.jpg', '_augmentation.npy'), augmentation_arr)
            progressbar.update(1)


def main():
    for mset in ['train', 'test']:
        imgs_path = glob(f"../datasets/aligned/96x112/{mset}/*.jpg")
        output_dir = f"../embeddings/sphereface/sphere20a_20171020/{mset}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = initialize_model()
        get_embedding(model, imgs_path, output_dir)


if __name__ == "__main__":
    main()
