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
    res = numpy.zeros(shape=(len(vocab)))
    res[vocab[word]] = 1.0
    return res


def get_batch(file, batch_size, words):
    res_v = []
    res_y = []
    for i in range(batch_size):
        # print(str(i)+". ")
        line = file.readline()
        if line == "":
            print("Done!")
            return res_v, res_y, True
        line = line.split(",")
        word = line[1]
        if word not in words:
            continue
        tag = line[2].replace("\n", "").replace("\r", "").split("\t")
        tag_prob = 1 / len(tag)
        tag_array = numpy.zeros(shape=len(tags))
        for tg in tag:
            if tg in tags:
                tag_array[tags.index(tg)] = tag_prob
        word_array = get_one_hot_rep(word, words)
        res_v.append(word_array)
        res_y.append(tag_array)
    return res_v, res_y, False


def get_topic_model():
    res = {}
    f = open("data/vectors.txt")
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        res[w[0]] = w[1].split(",")
    f.close()
    return res


def main(_):
    words = get_distinct_words()
    # Create the model
    vocab_size = len(words)

    # print("Topic model:")
    # print(get_topic_model())

    ff = open("data/words_no_topic.txt", "w")
    ff2 = open("data/words_with_topic.txt", "w")

    topic_models = get_topic_model()
    for key in topic_models:
        if key not in words.keys():
            print(key, file=ff)
        else:
            print(key, file=ff2)
    exit()

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
    tf.initialize_all_variables().run()

    f = open(file_path)

    cnt = 0
    eof = False
    while not eof:
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs, batch_ys , eof = get_batch(f, 10000 , words)
        n = numpy.array(batch_xs)
        sess.run(train_step, feed_dict={v: batch_xs, y_: batch_ys})
        cnt += 1
        print("Batch: "+str(cnt))
    f.close()

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # vv = []
    # for w in words:
    #     vv.append(get_one_hot_rep(w, words))
    #
    # res = sess.run(tf.nn.softmax(y), feed_dict={v: vv})
    # f = open("data/res.csv", "w")
    # for row in res:
    #     for num in row:
    #         f.write(str(num)+",")
    #     f.write("\n")
    # f.close()

    f = open("data/res.csv", "w")
    i = 0
    xx = []
    for w in words:
        print(i)
        i += 1
        if i % 1000 == 0:
            res = sess.run(tf.nn.softmax(y), feed_dict={v: xx})
            for row in res:
                for num in row:
                    f.write(str(num)+",")
                f.write("\n")
            xx = [get_one_hot_rep(w, words)]
        else:
            xx.append(get_one_hot_rep(w, words))
    res = sess.run(tf.nn.softmax(y), feed_dict={v: xx})
    for row in res:
        for num in row:
            f.write(str(num) + ",")
        f.write("\n")
    f.close()
    print("Done")

if __name__ == '__main__':
    tf.app.run(main=main)
