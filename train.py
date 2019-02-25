# @Time    : 2019/2/22 10:04
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : train.py
import time

import tensorflow as tf
import os
import imageio
import numpy as np
import shutil

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import  Flatten, Dropout, Dense, Convolution2D, \
    BatchNormalization, Activation


data_dir = r'D:\project\data-set\digits2'
moder_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_dir')
filename = 'train.tfrecords'
filenames = [filename]
HEIGTH, WIDTH, CHANNELS = (60, 160, 3)
BATCH_SIZE = 8
num_classes = 10
EPOCHS = 10


def remove_model_dir():
    if not os.path.exists(moder_dir):
        os.makedirs(moder_dir)
    shutil.rmtree(moder_dir)
    os.makedirs(moder_dir)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord():
    # 验证文件
    with tf.python_io.TFRecordWriter(filename) as writer:
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            img_arr = imageio.imread(img_path)
            image_raw = img_arr.tostring()
            label = int(img_name.split('_')[-1].split('.')[0])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())


def input_fn(data_path=r'D:\project\data-set\digits2', batch_size=BATCH_SIZE, is_training=True):
    def gen_generator():
        for i, img_name in enumerate(os.listdir(data_path)):
            img_path = os.path.join(data_path, img_name)
            img_arr = imageio.imread(img_path)
            img_arr = img_arr / 255.0
            label = int(img_name.split('_')[-1].split('.')[0])
            label = np.expand_dims(label, 0)
            yield img_arr, label
    dataset = tf.data.Dataset.from_generator(generator=gen_generator, output_types=(tf.float32, tf.int32),
                                             output_shapes=(tf.TensorShape([HEIGTH, WIDTH, CHANNELS]),tf.TensorShape([1])))

    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(buffer_size=1024)
    # dataset = dataset.map(parser, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(2)
    return dataset
    # iterator = dataset.make_one_shot_iterator()
    # feature, label = iterator.get_next()
    # with tf.Session() as sess:
    #     feature_, label_ = sess.run([feature, label])
    #     print(feature_)
    #     print(feature_.shape)
    #     print(label_)

def input_fn2(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    ds = ds.prefetch(2)
    # Return the dataset. (L)

    return ds


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_start_time = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_start_time)


def myloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def cnn_model():
    input_tensor = Input(shape=(HEIGTH, WIDTH, CHANNELS))
    x = input_tensor
    # 第一层卷积
    conv1 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    bath_normal1 = BatchNormalization()(conv1)
    dropout1 = Dropout(0.2)(bath_normal1)
    activate1 = Activation('relu')(dropout1)

    # 第二层卷积
    conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(activate1)
    bath_normal2 = BatchNormalization()(conv2)
    dropout2 = Dropout(0.2)(bath_normal2)
    activate2 = Activation('relu')(dropout2)

    # 第s三层卷积
    conv3 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(activate2)
    bath_normal3 = BatchNormalization()(conv3)
    dropout3 = Dropout(0.2)(bath_normal3)
    activate3 = Activation('relu')(dropout3)

    # flatten
    x = Flatten()(activate3)
    x = Dropout(0.25)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model


def train():
    remove_model_dir()
    tf.logging.set_verbosity(tf.logging.INFO)

    # model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=num_classes)
    model = cnn_model()
    model.summary()


    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    time_hist = TimeHistory()
    config = tf.estimator.RunConfig(model_dir=moder_dir, save_checkpoints_steps=100, log_step_count_steps=10, save_summary_steps=10)
    estimator = tf.keras.estimator.model_to_estimator(model, config=config)
    # estimator.train(input_fn=lambda: input_fn(), hooks=[time_hist])

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(is_training=True),
                                        max_steps=1000, hooks=[time_hist])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(is_training=False),steps= None)
    # train and evaluate modeld
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)

    total_time = sum(time_hist.times)
    avg_time_per_batch = np.mean(time_hist.times)
    print('total time use: {} seconds'.format(total_time))
    print('avg_time_per_batch: {} seconds'.format(avg_time_per_batch))
    print('{} images/seconds'.format(BATCH_SIZE/avg_time_per_batch))

if __name__ == '__main__':
    # write_tfrecord()
    train()