topic_vectors_path = "./data/vectors.txt"
one_gram_file_path = "./data/grams/1gram.csv"
one_gram_result_path = "./data/grams/1gram_final.csv"

vectors = {}

f = open(topic_vectors_path, encoding="ISO-8859-1")
lines = f.readlines()
for line in lines:
    w = line.split("\t")
    word = w[0]
    vectors[word] = 1
f.close()

f = open(one_gram_file_path)
fw = open(one_gram_result_path, mode='w')
lines = f.readlines()
for l in lines:
    sp = l.split(",")
    if sp[1] in vectors:
        print(l, file=fw, end='')
f.close()
fw.close()

print("Done")
