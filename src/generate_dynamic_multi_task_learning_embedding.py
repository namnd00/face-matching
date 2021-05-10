import re

import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
from tqdm import *
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


def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_dir = "../models/Dynamic_multi-task-learning/pretrained_models/20180115-025629_model"
            meta_file, ckpt_file = get_model_filenames(model_dir)
            load_model(model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')

            embedding_size = embeddings.get_shape()[1]

            for mset in ['train', 'test']:
                output_dir = '../embeddings/dynamic_multi_task_learning/%s/%s' % (model_dir.split('/')[-1], mset)
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
                        feed_dict = {images_placeholder: img, phase_train_placeholder: False,
                                     keep_probability_placeholder: True}
                        embed = sess.run(embeddings, feed_dict=feed_dict)
                        np.save(output_dir + '/%s.npy' % fn, embed)

                        augmentation_arr = np.array([], dtype=np.float32).reshape(0, embedding_size)
                        for i in range(100):
                            img_aug = seq.augment_image(img_org)
                            img_aug = prewhiten(img_aug)
                            img_aug = np.expand_dims(img_aug, axis=0)
                            feed_dict = {images_placeholder: img_aug, phase_train_placeholder: False,
                                         keep_probability_placeholder: True}
                            embed = sess.run(embeddings, feed_dict=feed_dict)

                            augmentation_arr = np.vstack((augmentation_arr, embed.reshape(1, embedding_size)))
                        np.save(output_dir + '/%s_augmentation.npy' % fn, augmentation_arr)


if __name__ == '__main__':
    main()
