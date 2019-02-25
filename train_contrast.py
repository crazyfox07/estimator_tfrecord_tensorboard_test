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
import random
from tensorflow.python.keras.layers import  Flatten, Dropout, Dense, Convolution2D, \
    BatchNormalization, Activation
from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras import backend as K

data_dir = r'D:\project\data-set\digits2'
moder_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_dir2')
filename = 'train.tfrecords'
filenames = [filename]
HEIGTH, WIDTH, CHANNELS = (60, 160, 3)
BATCH_SIZE = 32
num_classes = 10


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

def input_fn():
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [HEIGTH, WIDTH, CHANNELS])
        label = tf.cast(parsed['label'], tf.int32)
        label = tf.one_hot(label, num_classes)
        return image, label

    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(parser, num_parallel_calls=2)
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


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_start_time = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_start_time)


imgs = os.listdir(data_dir)
imgs_num = len(imgs)


def get_next_batch(batch_size=BATCH_SIZE, img_start=0, img_end=0):

    def one_hot_encode(num):
        arr = np.zeros(shape=(num_classes,))
        arr[num] = 1
        return arr

    items = imgs[img_start: img_end]
    train_x = np.zeros(shape=(batch_size, HEIGTH, WIDTH, CHANNELS))
    train_y = np.zeros(shape=(batch_size, 1))
    for i, item in enumerate(items):
        img_path = os.path.join(data_dir, item)
        img_arr = imageio.imread(img_path)
        img_arr = img_arr / 255
        label = int(item.split('_')[-1].split('.')[0])
        train_x[i] = img_arr
        train_y[i] = label#one_hot_encode(label)
    return train_x, train_y

# def cnn_model():
#     HIDDEN_NUM = 32
#     kernel_size = (3, 3)
#     pool_size = (2, 2)
#     input_tensor=Input(shape=(HEIGTH, WIDTH, CHANNELS))
#     x=input_tensor
#
#     #第一层卷积
#     x=Conv2D(HIDDEN_NUM, kernel_size,strides=(1, 1), padding='same', activation='relu')(x)
#     x=MaxPooling2D(pool_size, padding='valid')(x)
#     # 第二层卷积
#     x = Conv2D(HIDDEN_NUM*2, kernel_size, strides=(1, 1), padding='same', activation='relu')(x)
#     x = MaxPooling2D(pool_size, padding='valid')(x)
#     # 第三层卷积
#     x = Conv2D(HIDDEN_NUM*4, kernel_size, strides=(1, 1), padding='same', activation='relu')(x)
#     x = MaxPooling2D(pool_size, padding='valid')(x)
#
#     x=Flatten()(x)
#     x = Dropout(0.25)(x)
#     x = Dense(num_classes, activation='softmax')(x)
#     model = tf.keras.Model(inputs=input_tensor, outputs=x)
#     return model

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

    model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=(HEIGTH, WIDTH, CHANNELS), classes=num_classes)
    # model = cnn_model()
    model.summary()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    start_time = time.time()
    times = []
    for epoch in range(30):
        step = 0
        while True:
            img_start = BATCH_SIZE * step
            img_end = BATCH_SIZE * (step + 1)
            X_train, Y_train = get_next_batch(batch_size=BATCH_SIZE,img_start=img_start, img_end=img_end)
            # 训练模型
            outputs = model.train_on_batch(X_train, Y_train)

            if step % 1 == 0:

                end_time = time.time()
                print('step: {},  outputs: {}, time use: {}'.format(step, outputs, end_time-start_time))
                times.append(end_time-start_time)
                start_time = end_time
            step += 1
            if img_end >= imgs_num:
                print('train over')
                total_time = sum(times)
                avg_time_per_batch = np.mean(times)
                print('total time use: {} seconds'.format(total_time))
                print('avg_time_per_batch: {} seconds'.format(avg_time_per_batch))
                print('{} images/seconds'.format(BATCH_SIZE / avg_time_per_batch))
                break


if __name__ == '__main__':
    # write_tfrecord()
    train()
    # get_next_batch()