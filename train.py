import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
import os
import numpy as np
from model import GANomaly
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

from absl import app
from absl import flags
from absl import logging

DATASET_PATH = '../datasets/MachineLearningCVE/'
DATASET_EXTENSION = '.csv'

FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 300, "batch_size")
flags.DEFINE_integer("isize", 78, "input size")
flags.DEFINE_string("ckpt_dir", 'ckpt', "checkpoint folder")
flags.DEFINE_integer("nz", 100, "latent dims")
flags.DEFINE_integer("nc", None, "input channels")
flags.DEFINE_integer("ndf", 64, "number of discriminator's filters")
flags.DEFINE_integer("ngf", 64, "number of generator's filters")
flags.DEFINE_integer("extralayers", 0, "extralayers for both G and D")
flags.DEFINE_list("encdims", None, "Layer dimensions of the encoder and in reverse of the decoder."
                                   "If given, dense encoder and decoders are used.")
flags.DEFINE_integer("niter", 20, "number of training epochs")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", None, "name of dataset")
DATASETS = ['mnist', 'cifar10', 'cic']
flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))
flags.DEFINE_integer("anomaly", None, "the anomaly idx")
flags.mark_flag_as_required('anomaly')
flags.mark_flag_as_required('isize')
flags.mark_flag_as_required('nc')

'''
def batch_resize(imgs, size: tuple):
    img_out = np.empty((imgs.shape[0], ) + size)
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out
'''

def main(_):
    opt = FLAGS
    # logging
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        logging.get_absl_handler().use_absl_log_file(FLAGS.dataset, log_dir=FLAGS.log_dir)

    files = sorted([k for k in os.listdir(DATASET_PATH) if k.endswith(DATASET_EXTENSION)])
    sets = [pd.read_csv(DATASET_PATH + file) for file in files]
    frames = pd.concat(sets, axis=0, ignore_index=True)
    columns = frames.columns.to_list()
    label = columns[-1]
    frames = frames.replace([-np.inf, np.inf], np.nan)
    frames = frames.dropna(axis=0)

    frames.loc[frames[label] == 'Web Attack � Brute Force', label] = 'Brute Force'
    frames.loc[frames[label] == 'Web Attack � XSS', label] = 'XSS'
    frames.loc[frames[label] == 'Web Attack � Sql Injection', label] = 'Sql Injection'

    dict_classes = {
        'BENIGN': 'benign',
        'DoS Hulk': 'dropped',
        'PortScan': 'attack',
        'DDoS': 'dropped',
        'DoS GoldenEye': 'dropped',
        'FTP-Patator': 'dropped',
        'SSH-Patator': 'dropped',
        'DoS slowloris': 'dropped',
        'DoS Slowhttptest': 'dropped',
        'Bot': 'dropped',
        'Brute Force': 'dropped',
        'XSS': 'dropped',
        'Infiltration': 'dropped',
        'Sql Injection': 'dropped',
        'Heartbleed': 'dropped'
    }

    new_label = [dict_classes[i] if i in dict_classes else i for i in frames[label]]
    frames.loc[:, label] = new_label
    classes_drop = ['dropped']
    frames = frames[~frames[label].isin(classes_drop)]
    labels = sorted(frames[label].unique().tolist())
    print(frames[label].value_counts())
    print(labels)

    X = frames.drop(label, axis=1)
    y = frames[label].to_frame()

    features = X.columns.to_list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
    scaler = MinMaxScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.transform(X_test[features])

    indices_to_remove = y_train[y_train[label] == 'attack'].index
    print('Indices to remove train: ' + str(len(indices_to_remove)))
    X_train_be = X_train.drop(indices_to_remove)
    y_train_be = y_train.drop(indices_to_remove)
    X_train_be.reset_index(drop=True, inplace=True)
    y_train_be.reset_index(drop=True, inplace=True)
    X_train_at = X_train.loc[indices_to_remove]
    y_train_at = y_train.loc[indices_to_remove]
    X_train_at.reset_index(drop=True, inplace=True)
    y_train_at.reset_index(drop=True, inplace=True)

    # Initialize the binarizer with the known classes

    lb = LabelBinarizer()
    lb.fit(labels)

    # One-hot encode the test labels
    y_train_be = lb.transform(y_train_be[label])
    y_train_at = lb.transform(y_train_at[label])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_be, y_train_be))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_train_at, y_train_at))
    train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
        opt.batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

    # training
    ganomaly = GANomaly(opt,
                        train_dataset,
                        valid_dataset=None,
                        test_dataset=test_dataset)
    ganomaly.fit(opt.niter)

    # evaluating
    ganomaly.evaluate_best(test_dataset)

'''
def main(_):
    opt = FLAGS
    # logging
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        logging.get_absl_handler().use_absl_log_file(FLAGS.dataset, log_dir=FLAGS.log_dir)
    # dataset
    if opt.dataset=='mnist':
        data_train, data_test = tf.keras.datasets.mnist.load_data()
    elif opt.dataset=='cifar10':
        data_train, data_test = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementError
    x_train, y_train = data_train
    x_test, y_test = data_test
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.reshape([-1,])
    y_test = y_test.reshape([-1,])
    # resize to (32, 32)
    if opt.dataset=='mnist':
        x_train = batch_resize(x_train, (32, 32))[..., None]
        x_test = batch_resize(x_test, (32, 32))[..., None]
    # normalization
    mean = x_train.mean()
    stddev = x_train.std()
    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev
    logging.info('{}, {}'.format(x_train.shape, x_test.shape))
    # define abnoraml data and normal
    # training data only contains normal
    x_train = x_train[y_train != opt.anomaly, ...]
    y_train = y_train[y_train != opt.anomaly, ...]
    y_test = (y_test == opt.anomaly).astype(np.float32)
    # tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
        opt.batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

    # training
    ganomaly = GANomaly(opt,
                        train_dataset,
                        valid_dataset=None,
                        test_dataset=test_dataset)
    ganomaly.fit(opt.niter)

    # evaluating
    ganomaly.evaluate_best(test_dataset)
'''

if __name__ == '__main__':
    app.run(main)