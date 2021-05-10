import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import *
from imgaug import augmenters as iaa
from glob import glob
import sys

sys.path.insert(0, "../models/facenet/src")
import facenet

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


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def main():
    pretrained_models = glob("../models/facenet/pretrained_models/*")
    for model in pretrained_models:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                for mset in ['train', 'test']:
                    output_dir = '../embeddings/facenet/%s/%s' % (model.split('\\')[-1], mset)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    for rdir, sdir, files in os.walk(f'../datasets/aligned/160x160/{mset}'):
                        for file in tqdm(files):
                            if '.jpg' not in file:
                                continue
                            fn, fe = os.path.splitext(file)
                            img_path = os.path.join(rdir, file)
                            img_org = cv2.imread(img_path)
                            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

                            img = prewhiten(img_org)
                            img = np.expand_dims(img, axis=0)
                            feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                            embed = sess.run(embeddings, feed_dict=feed_dict)
                            np.save(output_dir + '/%s.npy' % fn, embed)

                            augmentation_arr = np.array([], dtype=np.float32).reshape(0, 512)
                            for i in range(100):
                                img_aug = seq.augment_image(img_org)
                                img_aug = prewhiten(img_aug)
                                img_aug = np.expand_dims(img_aug, axis=0)
                                feed_dict = {images_placeholder: img_aug, phase_train_placeholder: False}
                                embed = sess.run(embeddings, feed_dict=feed_dict)

                                augmentation_arr = np.vstack((augmentation_arr, embed.reshape(1, 512)))
                            np.save(output_dir + '/%s_augmentation.npy' % fn, augmentation_arr)


if __name__ == '__main__':
    main()
