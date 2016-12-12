from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy

FLAGS = None

file_path = "./data/grams/1gram.csv"
tags = ["algorithm", "android", "annotations", "ant", "apache", "applet", "arraylist", "arrays", "awt", "c#", "c++", "class", "collections", "concurrency", "database", "date", "design-patterns", "eclipse", "encryption", "exception", "file-io", "file", "generics", "google-app-engine", "gwt", "hadoop", "hashmap", "hibernate", "html", "http", "image", "inheritance", "intellij-idea", "io", "jar", "java-ee", "java", "javafx", "javascript", "jaxb", "jboss", "jdbc", "jersey", "jframe", "jni", "jpa", "jpanel", "jquery", "jsf", "json", "jsp", "jtable", "junit", "jvm", "libgdx", "linux", "list", "log4j", "logging", "loops", "maven", "methods", "multithreading", "mysql", "netbeans", "nullpointerexception", "object", "oop", "oracle", "osx", "parsing", "performance", "php", "python", "reflection", "regex", "rest", "scala", "security", "selenium", "serialization", "servlets", "soap", "sockets", "sorting", "spring-mvc", "spring-security", "spring", "sql", "sqlite", "string", "struts2", "swing", "swt", "tomcat", "unit-testing", "user-interface", "web-services", "windows", "xml"]


def get_distinct_words(vocab_size=65536):
    res = dict()
    with open(file_path) as f:
        lines = f.readlines()
        for l in lines:
            sp = l.split(",")
            if sp[1] in res:
                res[sp[1]] += 1
            else:
                res[sp[1]] = 1
    tmp = sorted(res.items(), key=lambda x: x[1], reverse=True)
    tmp = tmp[:vocab_size]
    final_res = dict()
    for t, _ in tmp:
        final_res[t] = len(final_res)
    return final_res


def get_one_hot_rep(word, vocab):
    res = numpy.zeros(shape=(len(vocab)), dtype="int32")
    res[vocab[word]] = 1
    return res


def get_batch(file, batch_size):
    for i in range(batch_size):
        print(str(i)+". ")
        line = file.readline()
        if line == "":
            print("Done!")
            break
        line = line.split(",")
        word = line[1]
        tag = line[2].replace("\n", "").replace("\r", "").split("\t")
        tag_prob = 1 / len(tag)
        tag_array = numpy.zeros(shape=len(tags))
        for tg in tag:
            if tg in tags:
                tag_array[tags.index(tg)] = tag_prob
        word_array = get_one_hot_rep(word, words)



words = get_distinct_words()
print(get_one_hot_rep('you', words))

f = open(file_path)

get_batch(f, 1000000)

f.close()

exit()




def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    vocab_size = 1000
    # Create the model
    v = tf.placeholder(tf.float32, [None, vocab_size])
    wp = tf.Variable(tf.zeros([vocab_size, 300]))
    wc = tf.Variable(tf.zeros([300, 100]))
    b = tf.Variable(tf.zeros([100]))
    y = tf.matmul(tf.matmul(v, wp), wc) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 100])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={v: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
