import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


train_dir = './training_dataset/out'


husky = []
label_husky = []
kiwawa = []
label_kiwawa = []


def get_files(file_dir, ratio):
    for file in os.listdir(train_dir):
        if file.endswith('0.jpg'):
            husky.append(file_dir + file)
            label_husky.append(0)
        elif file.endswith('1.jpg'):
            kiwawa.append(file_dir + file)
            label_kiwawa.append(1)

    image_list = np.hstack((husky, kiwawa))
    label_list = np.hstack((label_husky, label_kiwawa))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  # validation
    n_train = n_sample - n_val  # train

    t_images = all_image_list[0:n_train]
    t_labels = all_label_list[0:n_train]
    for i in t_labels:
        t_labels = [int(float(i))]
    v_images = all_image_list[n_train:-1]
    v_labels = all_label_list[n_train:-1]
    for i in v_labels:
        v_labels = [int(float(i))]

    return t_images, t_labels, v_images, v_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

