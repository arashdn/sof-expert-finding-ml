one_gram_file_path = "./data/grams/1gram_final.csv"
res_file_path = "./data/doc_len.txt"

res = {}

f = open(one_gram_file_path, encoding="ISO-8859-1")
lines = f.readlines()
for line in lines:
    w = line.split(",")
    doc_id = w[0]
    if doc_id in res:
        res[doc_id] += 1
    else:
        res[doc_id] = 1
f.close()

res_max = res[max(res, key=res.get)]


f = open(res_file_path,mode='w')

for key in res:
    print(key+":"+str(res[key]/res_max), file=f)


f.close()
