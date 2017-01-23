import tensorflow as tf
import numpy

topic_vectors_path = "./data/vectors.txt"
doc_len_path = "./data/doc_len.txt"
one_gram_file_path = "./data/grams/1gram_final.csv"


tags = ["algorithm", "android", "annotations", "ant", "apache", "applet", "arraylist", "arrays", "awt", "c#", "c++", "class", "collections", "concurrency", "database", "date", "design-patterns", "eclipse", "encryption", "exception", "file-io", "file", "generics", "google-app-engine", "gwt", "hadoop", "hashmap", "hibernate", "html", "http", "image", "inheritance", "intellij-idea", "io", "jar", "java-ee", "java", "javafx", "javascript", "jaxb", "jboss", "jdbc", "jersey", "jframe", "jni", "jpa", "jpanel", "jquery", "jsf", "json", "jsp", "jtable", "junit", "jvm", "libgdx", "linux", "list", "log4j", "logging", "loops", "maven", "methods", "multithreading", "mysql", "netbeans", "nullpointerexception", "object", "oop", "oracle", "osx", "parsing", "performance", "php", "python", "reflection", "regex", "rest", "scala", "security", "selenium", "serialization", "servlets", "soap", "sockets", "sorting", "spring-mvc", "spring-security", "spring", "sql", "sqlite", "string", "struts2", "swing", "swt", "tomcat", "unit-testing", "user-interface", "web-services", "windows", "xml"]
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


def get_initial_weights_for_tags_matrix():
    weights = []
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
            return len(doc_p) *1.0,doc_p,res_v, res_y, True
        line = line.split(",")
        docID= int(line[0])
        voteShareval = []
        voteShareval.append(docs_len[docID])
        word = line[1]
        if word not in words:
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
        doc_p.append(voteShareval)
        # bs[0][0] = len(doc_p) *1.0
        bs = len(doc_p) *1.0
    return bs,doc_p,res_v, res_y, False


def get_file_len():
    with open(one_gram_file_path) as f:
        return sum(1 for _ in f)


def main(_):
    initial_words = get_initial_words()
    docs_len = load_doc_len_normalizer()
    word_wordIndex,wordIndex_word = get_distinct_words(initial_words)

    assert wordIndex_word[word_wordIndex['file']] == 'file'

    wordIndex_vector = get_topic_model(word_wordIndex)

    assert str(wordIndex_vector[word_wordIndex['file']][6]) == '0.1193'  # loaded from mallet file

    initial_w = get_initial_weights_for_tags_matrix()
    vocab_size = len(word_wordIndex)
    wordIndex_vector_array = convert_topic_dict_to_matrix(wordIndex_vector)

    assert wordIndex_vector_array[word_wordIndex['file']] == wordIndex_vector[word_wordIndex['file']]

    print("Data Pre-processing Completed")

    print("Number of words in file: ", end='')
    file_len = get_file_len()
    print(file_len)

    batch_cnt = file_len//WORD_PER_BATCH
    print("Total batch: "+str(batch_cnt))

    v = tf.placeholder(tf.float32, [None, vocab_size]) # a matrix, each row is one-hot representation of words
    doc_importance = tf.placeholder(tf.float32, [None, 1])
    batch_size = tf.placeholder(tf.float32, numpy.zeros(shape=(0,1)))  # ?????
    wp = tf.Variable(wordIndex_vector_array)
    wc = tf.Variable(initial_w)
    b = tf.Variable(tf.zeros([100]))
    y_logit = tf.exp(tf.matmul(tf.matmul(v, wp), wc) + b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 100])

    param1 = 1.0 / batch_size
    param2 = (lambda_regularization / (2 * batch_size))

    cross_entropy = param1* tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logit, y_)) + param2* (tf.nn.l2_loss(wp) + tf.nn.l2_loss(wc))

    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
    train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()

    f = open(one_gram_file_path)

    cnt = 0
    eof = False
    count_iter = 100

    while not eof:
        bs, docp, batch_xs, batch_ys, eof = get_batch(f, WORD_PER_BATCH, word_wordIndex, docs_len)
        # n = numpy.array(batch_xs)
        if len(batch_xs) == 0:
            continue

        for ii in range(0, count_iter):

            sess.run(train_step, feed_dict={v: batch_xs, y_: batch_ys, doc_importance: docp, batch_size: bs})

            avg_cost = sess.run(cross_entropy, feed_dict={v: batch_xs, y_: batch_ys, doc_importance: docp, batch_size: bs})
            sum_wp = sess.run(tf.reduce_sum(wp))
            sum_wc = sess.run(tf.reduce_sum(wc))
            sum_bias = sess.run(tf.reduce_sum(b))

            print("Batch and Run: " + str(cnt) + "__" + str(ii) + "  " + str(avg_cost) + " wp = " + str(
                sum_wp) + " wc = " + str(
                sum_wc) + " sum_bias = " + str(sum_bias))

        cnt += 1
        print("Batch: " + str(cnt) + "/" + str(batch_cnt))

    f.close()

if __name__ == '__main__':
    tf.app.run(main=main)