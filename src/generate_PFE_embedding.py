import os
import numpy as np
import tensorflow as tf
import argparse
import cv2
import imp

from tqdm import tqdm
from imgaug import augmenters as iaa

import sys
sys.path.insert(0, "../models/Probabilistic_Face_Embeddings")
from utils.tflib import mutual_likelihood_score_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                                   allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def initialize(self, config, num_classes=None):
        '''
            Initialize the graph from scratch according to config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images')
                self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # Initialialize the backbone network
                network = imp.load_source('embedding_network', config.embedding_network)
                mu, conv_final = network.inference(self.images, config.embedding_size)

                # Initialize the uncertainty module
                uncertainty_module = imp.load_source('uncertainty_module', config.uncertainty_module)
                log_sigma_sq = uncertainty_module.inference(conv_final, config.embedding_size,
                                                            phase_train=self.phase_train,
                                                            weight_decay=config.weight_decay,
                                                            scope='UncertaintyModule')

                self.mu = tf.identity(mu, name='mu')
                self.sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')

                # Build all losses
                loss_list = []
                self.watch_list = {}

                MLS_loss = mutual_likelihood_score_loss(self.labels, mu, log_sigma_sq)
                loss_list.append(MLS_loss)
                self.watch_list['loss'] = MLS_loss

                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                loss_list.append(reg_loss)
                self.watch_list['reg_loss'] = reg_loss

                total_loss = tf.add_n(loss_list, name='total_loss')
                grads = tf.gradients(total_loss, self.trainable_variables)

                # Training Operaters
                train_ops = []

                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                apply_gradient_op = opt.apply_gradients(list(zip(grads, self.trainable_variables)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Collect TF summary
                for k, v in self.watch_list.items():
                    tf.summary.scalar('losses/' + k, v)
                tf.summary.scalar('learning_rate', self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=99)

        return

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            self.images = self.graph.get_tensor_by_name('images:0')
            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.mu = self.graph.get_tensor_by_name('mu:0')
            self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))

    def extract_feature(self, images):
        feed_dict = {self.images: images,
                     self.phase_train: False,
                     self.keep_prob: 1.0}
        mu = self.sess.run(self.mu, feed_dict=feed_dict)

        return mu


def parse_argument():
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--model', default='../models/Probabilistic_Face_Embeddings/pretrained_models', help='path to load model.')

    return parser.parse_args()


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


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


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            pretrained_models = [os.path.join(args.model, p) for p in os.listdir(args.model)]
            for model in pretrained_models:
                print(model)
                network = Network()
                network.load_model(model)

                for mset in ['train', 'test']:
                    output_dir = '../embeddings/PFE/%s/%s' % (model.split("\\")[-1], mset)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    for rdir, sdir, files in os.walk(f'../datasets/aligned/96x112/{mset}'):
                        for file in tqdm(files):
                            if '.jpg' not in file:
                                continue
                            fn, fe = os.path.splitext(file)
                            img_path = os.path.join(rdir, file)
                            img_org = cv2.imread(img_path)
                            img = prewhiten(img_org)
                            img = np.expand_dims(img, axis=0)
                            mu = network.extract_feature(img)
                            np.save(output_dir + '/%s.npy' % fn, mu)

                            augmentation_arr = np.array([], dtype=np.float32).reshape(0, 512)
                            for i in range(100):
                                img_aug = seq.augment_image(img_org)
                                img_aug = prewhiten(img_aug)
                                img_aug = np.expand_dims(img_aug, axis=0)
                                mu_aug = network.extract_feature(img_aug)
                                augmentation_arr = np.vstack((augmentation_arr, mu_aug.reshape(1, 512)))

                            np.save(output_dir + '/%s_augmentation.npy' % fn, augmentation_arr)


if __name__ == '__main__':
    main(parse_argument())
