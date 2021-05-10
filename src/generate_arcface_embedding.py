import torch
import cv2
import numpy as np
import os
import imp

from glob import glob
from tqdm import tqdm
from imgaug import augmenters as iaa


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, model_root, output_dir, tta=True):
    # pre-requisites
    assert (os.path.exists(img_root))
    output = os.path.join(output_dir, img_root.split("\\")[-1])
    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a = int((128 - 112) / 2)  # x start
    b = int((128 - 112) / 2 + 112)  # x end
    c = int((128 - 112) / 2)  # y start
    d = int((128 - 112) / 2 + 112)  # y end
    ccropped = resized[a:b, c:d]  # center crop the image
    ccropped = ccropped[..., ::-1]  # BGR to RGB

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    # load backbone from a checkpoint
    back_bone = backbone.IR_50((112, 112))
    back_bone.load_state_dict(torch.load(model_root))
    back_bone.to(device)

    # extract features
    back_bone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = back_bone(ccropped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(back_bone(ccropped.to(device)).cpu())

    embedding_size = 512
    augmentation_arr = np.array([], dtype=np.float32).reshape(0, embedding_size)
    for i in range(100):
        # augment images
        img_aug = seq.augment_image(resized)
        img_aug = img_aug[a:b, c:d]  # center crop the image
        img_aug = img_aug[..., ::-1]
        img_aug = img_aug.swapaxes(1, 2).swapaxes(0, 1)
        img_aug = np.reshape(img_aug, [1, 3, 112, 112])
        img_aug = np.array(img_aug, dtype=np.float32)
        img_aug = (img_aug - 127.5) / 128.0
        img_aug = torch.from_numpy(img_aug)

        with torch.no_grad():
            if tta:
                emb_augment = back_bone(img_aug.to(device)).cpu()
                features_augment = l2_norm(emb_augment)
            else:
                features_augment = l2_norm(back_bone(img_aug.to(device)).cpu())
        augmentation_arr = np.vstack((augmentation_arr, features_augment.reshape(1, embedding_size)))

    np.save(output.replace(".jpg", ".npy"), features)
    np.save(output.replace(".jpg", "_augmentation.npy"), augmentation_arr)


def main():
    model_dir = glob("../models/face.evoLVe.PyTorch/pretrained_models/*/*.pth")
    backbone = imp.load_source("backbone", "../models/face.evoLVe.PyTorch/pretrained_models/bh-ir50/model_irse.py")

    for mset in ['train', 'test']:
        img_dir = glob(f'../datasets/aligned/112x112/{mset}/*.jpg')

        for model in model_dir:
            print(model)
            name_model = model.split("\\")[-1].split('.pth')[0]
            output_dir = f'../embeddings/face.evoLVe.PyTorch/{name_model}/{mset}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for img in tqdm(img_dir):
                extract_feature(img, backbone, model, output_dir, True)


if __name__ == "__main__":
    main()
