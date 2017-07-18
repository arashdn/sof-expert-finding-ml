import tensorflow as tf
import numpy
import time
from pathlib import Path

from numpy.core.multiarray import dtype

wc_load = numpy.load("./save/wc_N30_bias.npy")
b_load = numpy.loadtxt("./save/b_N30_bias.txt")




topic_vectors_path = "./data/vectors.txt"
doc_len_path = "./data/doc_len.txt"
one_gram_file_path = "./data/grams/1gram_final.csv"
error_file_path = "./data/error_file.txt"
TF_file_path = "./data/grams/TF.csv"
IDF_file_path = "./data/grams/TFIDF.csv"
LAST_BATCH_path = "./save/last_batch.txt"




##java
# tags = ["algorithm", "android", "annotations", "ant", "apache", "applet", "arraylist", "arrays", "awt", "c#", "c++", "class", "collections", "concurrency", "database", "date", "design-patterns", "eclipse", "encryption", "exception", "file-io", "file", "generics", "google-app-engine", "gwt", "hadoop", "hashmap", "hibernate", "html", "http", "image", "inheritance", "intellij-idea", "io", "jar", "java-ee", "java", "javafx", "javascript", "jaxb", "jboss", "jdbc", "jersey", "jframe", "jni", "jpa", "jpanel", "jquery", "jsf", "json", "jsp", "jtable", "junit", "jvm", "libgdx", "linux", "list", "log4j", "logging", "loops", "maven", "methods", "multithreading", "mysql", "netbeans", "nullpointerexception", "object", "oop", "oracle", "osx", "parsing", "performance", "php", "python", "reflection", "regex", "rest", "scala", "security", "selenium", "serialization", "servlets", "soap", "sockets", "sorting", "spring-mvc", "spring-security", "spring", "sql", "sqlite", "string", "struts2", "swing", "swt", "tomcat", "unit-testing", "user-interface", "web-services", "windows", "xml"]
##php
tags = [".htaccess","ajax","android","apache","api","arrays","authentication","caching","cakephp","class","codeigniter","cookies","cron","css","csv","curl","database","date","datetime","doctrine","doctrine2","dom","drupal","email","encryption","facebook","facebook-graph-api","file","file-upload","foreach","forms","function","gd","get","html","html5","http","if-statement","image","include","java","javascript","joomla","jquery","json","laravel","laravel-4","linux","login","loops","magento","mod-rewrite","mongodb","multidimensional-array","mysql","mysqli","object","oop","pagination","parsing","paypal","pdf","pdo","performance","php","phpmyadmin","phpunit","post","preg-match","preg-replace","python","redirect","regex","rest","search","security","select","session","simplexml","soap","sorting","sql","sql-server","string","symfony2","table","twitter","upload","url","utf-8","validation","variables","web-services","wordpress","wordpress-plugin","xampp","xml","yii","zend-framework","zend-framework2"]
TAGS_LEN = len(tags)
TOPIC_LEN = 100
WORD_PER_BATCH = 1024

lambda_regularization = 0.01


def get_initial_words():
    res = {}
    # for each word we should apped it initial vector!
    f = open(topic_vectors_path,encoding = "ISO-8859-1")
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        word = w[0]
        res[word] = 1
    f.close()
    return res


def load_doc_len_normalizer():
    res = {}
    f = open(doc_len_path)
    lines = f.readlines()
    for line in lines:
        w = line.split(":")
        res[int(w[0])] = numpy.float32(w[1])
    f.close()
    return res


def get_distinct_words(initial_words, vocab_size=65536):  ## get word and return two indexes, first word->term frquncy then rank frequency -> word
    res = dict()
    countword = 0
    with open(one_gram_file_path) as f:
        lines = f.readlines()
        for l in lines:
            sp = l.split(",")
            countword += 1
            if sp[1] in res:
                res[sp[1]] += 1
            else:
                res[sp[1]] = 1
    tmp = sorted(res.items(), key=lambda x: x[1], reverse=True)
    tmp = tmp[:min(countword, vocab_size)]
    final_res = dict()
    for t, _ in tmp:
        final_res[t] = len(final_res)
    final_res2 = {v: k for k, v in final_res.items()}
    return final_res, final_res2


def get_array_from_csv_row(param):
    res = []
    doubles = param.split(",")
    for d in doubles:
        res.append(numpy.float32(d))
    return res  # read a comma seprated file and return it as an array


def get_topic_model(vocab):  # read mallet topic model and return a dict with word -> topic vector
    res = dict()
    res_out = dict()
    # for each word we should apped it initial vector!
    f = open(topic_vectors_path, encoding = "ISO-8859-1")
    lines = f.readlines()
    for line in lines:
        w = line.split("\t")
        word = w[0]
        word_array = get_array_from_csv_row(w[1])
        res[word] = word_array
    f.close()
    for ww in vocab:
        word_index = vocab[ww]
        vector_ww = res[ww]
        res_out[word_index] = vector_ww
    return res_out


def get_random_array_for_tag_init_weight():  # get initial weight for Wc with a uniform dist a 100(tags len) dim vector
    weight = []
    lower = -1 * numpy.math.sqrt(6.0 / TOPIC_LEN+TAGS_LEN)
    upper = +1 * numpy.math.sqrt(6.0 / TOPIC_LEN+TAGS_LEN)
    for i in range(TAGS_LEN):
        array = numpy.random.uniform(low=lower,high=upper)
        weight.append(array)
    return weight


def readbatchNumber(lastBatch_file):
    out = lastBatch_file.read_text()
    return int(out)


def load_wcLastBatch(batchnumber):
    wc = numpy.load(file='./save/wc_N'+str(batchnumber)+'.npy')
    return wc



def get_initial_weights_for_tags_matrix():
    weights = []
    lastBatch_file = Path(LAST_BATCH_path)
    if lastBatch_file.is_file():
        batchnumber = readbatchNumber(lastBatch_file)
        weights = load_wcLastBatch(batchnumber)
    else:
    # we sould append 100(tags len) array of length 100(topic len)
        for i in range(TOPIC_LEN):
            array = get_random_array_for_tag_init_weight()
            weights.append(array)
    return weights


def convert_topic_dict_to_matrix(wordIndex_vector):
    list_of_lists = []
    for index in range(0, len(wordIndex_vector)):
        list_of_lists.append(wordIndex_vector.get(index))
    return list_of_lists


def get_one_hot_rep(word, vocab):
    res = numpy.zeros(shape=(len(vocab)))
    res[vocab[word]] = 1.0
    return res


def get_batch(file, batch_size, words, docs_len):
    res_v = []
    res_y = []
    doc_p = []
    bs = numpy.zeros(shape=(1, 1))
    for i in range(batch_size):
        # print(str(i)+". ")
        line = file.readline()
        if line == "":
            print("Done!")
            return len(doc_p) * 1.0, doc_p, res_v, res_y, True
        line = line.split(",")
        docID= int(line[0])
        docs_len_val = []
        docs_len_val.append(docs_len[docID])
        word = line[1]
        if word not in words:
            file_error = open(error_file_path, mode='a')
            print("word: " + word + " ,not exist in file")
            file_error.close()
            continue
        tag = line[2].replace("\n", "").replace("\r", "").split("\t")
        tag_array = numpy.zeros(shape=len(tags))
        for tg in tag:
            if tg in tags:
                tag_array[tags.index(tg)] = 1.0
        tag_array = tag_array / sum(tag_array)
        word_array = get_one_hot_rep(word, words)
        res_v.append(word_array)
        res_y.append(tag_array)
        doc_p.append(docs_len_val)
        # bs[0][0] = len(doc_p) *1.0
        bs = len(doc_p) * 1.0
    return bs, doc_p, res_v, res_y, False


def get_file_len():
    with open(TF_file_path) as f:
        return sum(1 for _ in f)


def get_batch2(file, batch_size, words, wordIndex_vector):
    res_v = []
    res_y = []
    res_words = []
    bs = numpy.zeros(shape=(1, 1))
    for i in range(batch_size):
        line = file.readline()
        if line == "":
            print("Done!")
            return res_words, len(res_v) * 1.0, res_v, res_y, True
        line = line.split('\t')
        word= line[0]
        weights = line[1]
        if word not in words:
            file_error = open(error_file_path, mode='a')
            print("word: " + word + " ,not exist in file")
            file_error.close()
            continue
        tag_array= weights.split(',')
        tag_weights = []
        for i in range(0,len(tag_array)):
            tag_weights.append(float(tag_array[i]))
        summ = sum(tag_weights)
        if summ == 0:
            summ = 1
        tag_weights_normal = [x / summ for x in tag_weights]

        #word_array = get_one_hot_rep(word, words)

        res_v.append(wordIndex_vector[words[word]])
        res_y.append(tag_weights_normal)
        res_words.append(word)
        bs = len(res_v) * 1.0
    return res_words, bs, res_v, res_y, False



def main(_):
    print("Starting ...")
    initial_words = get_initial_words()
    # docs_len = load_doc_len_normalizer()
    word_wordIndex,wordIndex_word = get_distinct_words(initial_words)


    assert wordIndex_word[word_wordIndex['file']] == 'file'

    wordIndex_vector = get_topic_model(word_wordIndex)

    # java only
    # assert str(wordIndex_vector[word_wordIndex['file']][6]) == '0.1193'  # loaded from mallet file

    # initial_w = get_initial_weights_for_tags_matrix()
    vocab_size = len(word_wordIndex)
    wordIndex_vector_array = convert_topic_dict_to_matrix(wordIndex_vector)

    assert wordIndex_vector_array[word_wordIndex['file']] == wordIndex_vector[word_wordIndex['file']]

    print("Data Pre-processing Completed")


    v = tf.placeholder(tf.float32, [None, 100]) # a matrix, each row is one-hot representation of words
    batch_size = tf.placeholder(tf.float32)  # ?????
    wc = tf.constant(wc_load)
    b = tf.constant(b_load, dtype=tf.float32)
    y_logit = tf.add(tf.matmul(v, wc),b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 100])

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y_))
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()

    f_TF = open(TF_file_path)
    eof = False

    file_res = open("./save/final_res_bias.txt","w")
    print("word", file=file_res, end="\t")
    for t in tags:
        print(t,file=file_res, end="\t")
    print("\n",file=file_res,end='')

    while not eof:
        words, bs, batch_xs, batch_ys, eof = get_batch2(f_TF, WORD_PER_BATCH, word_wordIndex, wordIndex_vector)
        # n = numpy.array(batch_xs)
        if len(batch_xs) == 0:
            print("Zero batch size, skipped")
            continue
        if not eof:
            result = sess.run(tf.nn.softmax(y_logit), feed_dict={v: batch_xs})
            for ii in range(0,WORD_PER_BATCH):
                print(words[ii], end="\t", file=file_res)
                for c in result[ii]:
                    print(c, file=file_res, end="\t")

                print("\n",end='',file=file_res)

    f_TF.close()
    file_res.close()

if __name__ == '__main__':
    tf.app.run(main=main)