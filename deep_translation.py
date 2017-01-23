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
#file_path = "/media/lab/New Volume/LuceneSearch/data/grams/1gram.csv"
file_path = './data/grams/1gram.csv'
topic_vectors = "./data/vectors.txt"
vote_share_path = "./data/voteshare.txt"

one_gram_File =file_path
tags = ["algorithm", "android", "annotations", "ant", "apache", "applet", "arraylist", "arrays", "awt", "c#", "c++", "class", "collections", "concurrency", "database", "date", "design-patterns", "eclipse", "encryption", "exception", "file-io", "file", "generics", "google-app-engine", "gwt", "hadoop", "hashmap", "hibernate", "html", "http", "image", "inheritance", "intellij-idea", "io", "jar", "java-ee", "java", "javafx", "javascript", "jaxb", "jboss", "jdbc", "jersey", "jframe", "jni", "jpa", "jpanel", "jquery", "jsf", "json", "jsp", "jtable", "junit", "jvm", "libgdx", "linux", "list", "log4j", "logging", "loops", "maven", "methods", "multithreading", "mysql", "netbeans", "nullpointerexception", "object", "oop", "oracle", "osx", "parsing", "performance", "php", "python", "reflection", "regex", "rest", "scala", "security", "selenium", "serialization", "servlets", "soap", "sockets", "sorting", "spring-mvc", "spring-security", "spring", "sql", "sqlite", "string", "struts2", "swing", "swt", "tomcat", "unit-testing", "user-interface", "web-services", "windows", "xml"]
#finalWords =[]
#finalDict =dict()





def get_distinct_words(initial_words,vocab_size=65536):

    res = dict()
    countword = 0
    with open(one_gram_File) as f:
        lines = f.readlines()
        for l in lines:
            sp = l.split(",")
            if sp[1] in initial_words:
                countword+=1
                if sp[1] in res:
                   res[sp[1]] += 1
                else:
                   res[sp[1]] = 1
    tmp = sorted(res.items(), key=lambda x: x[1], reverse=True)
    tmp = tmp[:min(countword,vocab_size)]
    out =''

    for w in tmp:
        out += str(w)
        out +='\n'

    ff = open("./top_words_python.txt","w")
    print(out , file=ff)
    ff.close()
    final_res = dict()
    for t, _ in tmp:
        aa = len(final_res)
        final_res[t] = aa
    final_res2 = {v: k for k, v in final_res.items()}
    return final_res, final_res2


def get_one_hot_rep(word, vocab):
    res = numpy.zeros(shape=(len(vocab)))
    res[vocab[word]] = 1.0
    return res


def get_batch(file, batch_size, words, voteShare):
    res_v = []
    res_y = []
    doc_p = []
    #w, h = 1, 1
    bs = numpy.zeros(shape=(1,1))
    for i in range(batch_size):
        # print(str(i)+". ")
        line = file.readline()
        if line == "":
            print("Done!")
            return len(doc_p) *1.0,doc_p,res_v, res_y, True
        line = line.split(",")
        docID= int(line[0])
        voteShareval = []
        voteShareval.append(voteShare[docID])
        word = line[1]
        if word not in words:
            continue
        tag = line[2].replace("\n", "").replace("\r", "").split("\t")
        #tag_prob = 1.0 / len(tag)
        tag_array = numpy.zeros(shape=len(tags))
        for tg in tag:
            if tg in tags:
                tag_array[tags.index(tg)] = 1.0
        tag_array = tag_array / sum(tag_array)
        word_array = get_one_hot_rep(word, words)
        res_v.append(word_array)
        res_y.append(tag_array)
        doc_p.append(voteShareval)
        bs[0][0] = len(doc_p) *1.0
    return bs,doc_p,res_v, res_y, False


def getArray(param):
    res = []
    doubles = param.split(",")
    for d in doubles:
        res.append(numpy.float32(d))
    return res


def getWordIndex(word,vocab):
    if word in vocab.keys():
        return vocab[word]
    return -1


def get_topic_model(vocab):
    res = dict()
    res_out = dict()
    # for each word we should apped it initial vector!
    f = open(topic_vectors,encoding = "ISO-8859-1")
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        word = w[0]
        wordArray = getArray(w[1])
        res[word] = wordArray
    f.close()
    for ww in vocab:
        wordIndex = vocab[ww]
        vectorww = res[ww]
        res_out[wordIndex] = vectorww
    return res_out


def getRandomArray():
    weight = []
    lower = -1 * numpy.math.sqrt(6.0 / 200)
    upper = +1 * numpy.math.sqrt(6.0 / 200)
    for i in range(100):
        array = numpy.random.uniform(low=lower,high=upper)
        weight.append(array)
    return weight


def loadInitialWeights():
    weight = []
    # we sould append 100 array of length 100
    for i in range(100):
        array = getRandomArray()
        weight.append(array)
    return weight


def loadVoteShare():
    res = {}
    f = open(vote_share_path)
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        res[int(w[0])] = numpy.float32(w[1])
    f.close()
    return res


def get_initial_words():
    res = {}
    # for each word we should apped it initial vector!
    f = open(topic_vectors,encoding = "ISO-8859-1")
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        word = w[0]
        res[word] = 1
    return res


def getArrayVector(wordIndex_vector):
    list_of_lists = []
    for index in range(0, len(wordIndex_vector)):
        list_of_lists.append(wordIndex_vector.get(index))
    return list_of_lists


def getVocabOneHot(wordindex_word,word_WordIndex):
    xx = []
    for i in range(0,len(wordindex_word)):
        xx.append(get_one_hot_rep(wordindex_word[i], word_WordIndex))
    return xx


def printTopWords(phpwords, indexes, wordindexWord):
    out =''
    for ii in range(0, len(indexes)):
        out += '('+str(phpwords[indexes[ii]])+','+ str(wordindexWord[indexes[ii]]) + ')\t'
    print(out)


def main(_):

    initial_words = get_initial_words()
    voteShare = loadVoteShare()
    word_WordIndex,WordIndex_word  = get_distinct_words(initial_words)
    # Create the model
    print("Topic model:")
    #print(get_topic_model())
    wordIndex_vector = get_topic_model(word_WordIndex)
    #words = finalDict
    initialW = loadInitialWeights()
    vocab_size = len(word_WordIndex)
    wordIndex_vector_array = getArrayVector(wordIndex_vector)




    v = tf.placeholder(tf.float32, [None, vocab_size])
    docImportance = tf.placeholder(tf.float32,[None, 1])
    batchSize= tf.placeholder(tf.float32,numpy.zeros(shape=(0,0)))
    #wp = tf.Variable(tf.zeros([vocab_size, 100]))
    wp = tf.Variable(wordIndex_vector_array)
    #wc = tf.Variable(tf.zeros([100, 100]))
    wc = tf.Variable(initialW)
    b = tf.Variable(tf.zeros([100]))
    y = tf.exp(tf.matmul(tf.matmul(v, wp), wc) + b)

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
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    lambda_regularization = 0.01
    param2 = (lambda_regularization /(2* batchSize))
    #param2 = (lambda_regularization /(2* 1))
    param1 = 1.0/batchSize
    #param1 = 1.0/1

    #cross_entropy = param1* (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_) * docImportance))
    cross_entropy = param1* tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))\
        + param2* (tf.nn.l2_loss(wp) + tf.nn.l2_loss(wc))


    #train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()

    #printweights(sess, v, word_WordIndex, y, "./res0.csv")

    f = open(file_path)

    cnt = 0
    eof = False
    countiter = 10000
    #you_one_hot = [get_one_hot_rep('you', word_WordIndex)]
    php_one_hot = [get_one_hot_rep('php', word_WordIndex)]
    allhotWords = getVocabOneHot(WordIndex_word,word_WordIndex)

    while not eof:
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        bs, docp, batch_xs, batch_ys , eof = get_batch(f, 1024 , word_WordIndex,voteShare)
        #n = numpy.array(batch_xs)
        if len(batch_xs)==0:
            continue
        for ii in range(0,countiter):
            #sess.run(train_step, feed_dict={v: batch_xs, y_: batch_ys,docImportance:docp})
            sess.run(train_step, feed_dict={v: batch_xs, y_: batch_ys,docImportance:docp, batchSize:bs})

            avg_cost = sess.run(cross_entropy,
            #feed_dict={v: batch_xs, y_: batch_ys, docImportance: docp})
            feed_dict={v: batch_xs, y_: batch_ys, docImportance: docp, batchSize: bs})
            sum_wp = sess.run(tf.reduce_sum(wp))
            sum_wc = sess.run(tf.reduce_sum(wc))
            sum_bias = sess.run(tf.reduce_sum(b))

            print("Batch and Run: " + str(cnt) + "__" +str(ii)+"  " + str(avg_cost) + " wp = " + str(sum_wp) + " wc = " + str(
                sum_wc) + " sum_bias = " + str(sum_bias))

            if(ii> 400):
                print('here')
            res_php = sess.run(tf.nn.softmax(y), feed_dict={v: allhotWords})



            phpwords = [i[tags.index('php')] for i in res_php]
            indexes = numpy.argpartition(phpwords, -5)[-5:]
            printTopWords(phpwords,indexes,WordIndex_word)


            #res_php = sess.run(tf.nn.softmax(y), feed_dict={v: php_one_hot})
            #print("probability of php -->" + str(res_php[0]))




            #chist = numpy.argsort(phpwords[indexes])

            #print(WordIndex_word[topindexesPhP])
            print('______________________________')

            ii+=1


        print("")

        cnt += 1
        if cnt > 1:
            break
        print("Batch: "+str(cnt))

    f.close()

    #printweights(sess, v, word_WordIndex, y, "./res.csv")
    printweights('php',sess, v, word_WordIndex, y, "./res.csv")
    printMatrixes(sess, wp, wc, "./mats.csv")

def printMatrixes(sess,wp,wc, fileName):
    f = open(fileName+"_wp", "w")
    res = sess.run(wp)
    for row in res:
        for num in row:
            f.write(str(num) + ",")
        f.write("\n")
    f.close()

    f = open(fileName+"_wc", "w")
    res = sess.run(wc)
    for row in res:
        for num in row:
            f.write(str(num) + ",")
        f.write("\n")
    f.close()


    pass

def printweights(givenWord, sess, v, word_WordIndex, y, fileoutput):
    f = open(fileoutput, "w")
    xx = [get_one_hot_rep(givenWord, word_WordIndex)]
    res = sess.run(tf.nn.softmax(y), feed_dict={v: xx})
    for row in res:
        for num in row:
            f.write(str(num) + ",")
        f.write("\n")
    f.close()
    print("Done")







#    def printweights(givenWord, sess, v, word_WordIndex, y, fileoutput):
#    f = open(fileoutput, "w")
#    i = 0
#    xx = []
#    for w in word_WordIndex:
#        print(i)
#        i += 1
#        if i % 1000 == 0:
#            res = sess.run(tf.nn.softmax(y), feed_dict={v: xx})
#            for row in res:
#                for num in row:
#                    f.write(str(num) + ",")
#                f.write("\n")
#            xx = [get_one_hot_rep(w, word_WordIndex)]
#        else:
#            xx.append(get_one_hot_rep(w, word_WordIndex))
#    res = sess.run(tf.nn.softmax(y), feed_dict={v: xx})
#    for row in res:
#        for num in row:
#            f.write(str(num) + ",")
#        f.write("\n")
#    f.close()
#    print("Done")


if __name__ == '__main__':
    tf.app.run(main=main)
