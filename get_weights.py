topWords = {}
f = open("./data/top_words.txt")
for word in f.readlines():
    topWords[word.strip()] = 1
f.close()

f = open('./data/mallet/php_100_count.txt')
ff = open("./data/vectors.txt", "w")
res = {}
count = 0
count2 = 0
for line in f.readlines():
    print(str(count) + "/" + str(count2))
    count += 1
    s = line.split(' ')
    key = s[1]
    if key not in topWords.keys():
        continue
    count2 += 1
    res[key] = [0] * 100
    del s[0]
    del s[0]
    sum = 0
    for i in s:
        m = i.split(':')
        sum += int(m[1])
    if sum > 0:
        for i in s:
            m = i.split(':')
            res[key][int(m[0])] = float(m[1]) / sum
    s = key + "\t"
    for w in res[key]:
        s += str(w) + ","
    s = s[:-1]
    print(s, file=ff)
f.close()
ff.close()
