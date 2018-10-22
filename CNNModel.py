import os
from PIL import Image
import numpy as np
import tensorflow as tf

data_dir = "./training_dataset/out"
train = False
model_path = "./model/image_cnn_model"


# read image and label from folder
def read_data(data_dir):
    data = list()
    label = list()
    fpath = list()
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        fpath.append(path)
        image = Image.open(path)
        image_data = np.array(image) / 255
        image_label = int(fname.split("_")[0])
        data.append(image_data)
        label.append(image_label)

    data = np.array(data)
    label = np.array(label)

    print("shape of data: {}\tshape of label:{}".format(data.shape, label.shape))
    return fpath, data, label


fpath, data, label = read_data(data_dir)

num_class = len(set(label))

data_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
label_placeholder = tf.placeholder(tf.int32, [None])

# dropout: training=0.25 test=0
dropout_placeholder = tf.placeholder(tf.float32)

# define convolution layer 0
conv0 = tf.layers.conv2d(data_placeholder, 20, 5, activation=tf.nn.relu)
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# define convolution layer 1
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])


flatten = tf.layers.flatten(pool1)

# define fully connected layer
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

dropout_fc = tf.layers.dropout(fc, dropout_placeholder)

logits = tf.layers.dense(dropout_fc, num_class)

predicted_labels = tf.arg_max(logits, 1)

# define loss function
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(label_placeholder, num_class),
    logits=logits)

mean_loss = tf.reduce_mean(losses)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)

saver = tf.train.Saver()

with tf.Session() as sess:
    if train:
        print("Train")
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {
            data_placeholder: data,
            label_placeholder: label,
            dropout_placeholder: 0.25
        }
        for step in range(150):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("Training ends. The model has saved to{}".format(model_path))

    else:
        print("Test")
        saver.restore(sess, model_path)
        print("Loading model from{}".format(model_path))
        label_name_dict = {
            0:  "Husky",
            1: "Kiwawa",
        }

        test_feed_dict = {
            data_placeholder: data,
            label_placeholder: label,
            dropout_placeholder: 0
        }

        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # prediction = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # max_index = np.argmax(prediction)
        # if max_index == 0:
        #     print('This is a husky with possibility %.6f' % prediction[:, 0])
        # elif max_index == 1:
        #     print('This is a jiwawa with possibility %.6f' % prediction[:, 1])
        # img = get_one_image(val)  # 通过改变参数train or val，进而验证训练集或测试集
        # evaluate_one_image(img)

        for fp, real_label, predicted_label in zip(fpath, label, predicted_labels_val):
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(fp, real_label_name, predicted_label_name))








