import os
import tensorflow as tf
from PIL import Image

ori_image = './training_dataset/ori'

new_image = './training_dataset/out'

classes = {'husky', 'kiwawa'}

num_samples = 160


def create_record():
    writer = tf.python_io.TFRecordWriter("dog_train.tfrecords")
    for index, name in enumerate(classes):
        class_path = ori_image + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_raw = img.tobytes()
            print(index, img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    label = tf.cast(label, tf.int32)
    return img, label


if __name__ == '__main__':
    create_record()
    batch = read_and_decode('dog_train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_samples):
            example, lab = sess.run(batch)
            img = Image.fromarray(example, 'RGB')
            img.save(new_image + '/' + str(lab) + '_' + str(i) + 'samples' + '.jpg')
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()

