import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' WARNINGS and INFO messages are not printed
import tensorflow as tf
print(tf.__version__)
import numpy as np
from model import GANomaly
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

from absl import app
from absl import flags
from absl import logging

# DATASET_PATH = '../datasets/MachineLearningCVE/'
# DATASET_PATH = '../datasets/IDS-2017/'
DATASET_PATH = '../datasets/VOIP/'
DATASET_EXTENSION = '.csv'

FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("isize", None, "input size")
flags.DEFINE_string("ckpt_dir", './results/', "checkpoint folder")
flags.DEFINE_integer("niter", 6, "number of training epochs")
flags.DEFINE_float("lr", 1e-3, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", 'voip', "name of dataset")
DATASETS = ['voip']
flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))
# flags.mark_flag_as_required('isize')

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
    print(files)
    sets = [pd.read_csv(DATASET_PATH + file) for file in files]
    frames = pd.concat(sets, axis=0, ignore_index=True)
    columns = frames.columns.to_list()
    label = columns[-1]
    frames = frames.replace([-np.inf, np.inf], np.nan)
    frames = frames.dropna(axis=0)
    print(frames)

    columns_to_drop = ["id", "expiration_id", "src_ip", "src_mac", "src_oui", "src_port",
    "dst_ip", "dst_mac", "dst_oui", "vlan_id", "tunnel_id", "application_name",
    "application_category_name", "application_is_guessed", "application_confidence",
    "requested_server_name", "client_fingerprint", "server_fingerprint", "user_agent",
    "content_type", "udps.timestamp"]
    frames = frames.drop(columns_to_drop, axis=1)

    columns = frames.columns.to_list()
    FLAGS.isize = len(columns)-1
    print("The number of columns is: " + str(len(columns)-1))
    print("The label is: " + label)
    print(frames)
    labels = sorted(frames[label].unique().tolist())
    print('Final labels used: ' + ', '.join(labels))
    print(frames[label].value_counts())

    X = frames.drop(label, axis=1)
    y = frames[label].to_frame()
    features = X.columns.to_list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
    scaler = MinMaxScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.transform(X_test[features])

    indices_to_remove = y_train[y_train[label] == 'attack'].index
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
    # y_train_at = lb.transform(y_train_at[label])
    y_test = lb.transform(y_test[label])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_be, y_train_be))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
        opt.batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

    # training
    ganomaly = GANomaly(opt,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset)
    ganomaly.fit(opt.niter)

    # evaluating
    ganomaly.evaluate_best(test_dataset)

if __name__ == '__main__':
    app.run(main)