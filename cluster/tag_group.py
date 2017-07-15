
tag = 'java'

f = open("./data/res_"+tag+".csv")

res = {}
f.readline()
for line in f:
    s = line.replace('"','').strip().split(',')
    if s[1] in res.keys():
        res[s[1]].append(s[0])
    else:
        res[s[1]] = [s[0]]
    # print(s)
f.close()

f = open("./data/cluster_"+tag+"_final.csv","w")
found_tags = []
for r in res:
    s = ""
    for k in res[r]:
        found_tags.append(k)
        s = s + k +","
    print(res[r])
    s = s[:-1]
    print(s, file=f, sep='\r\n')
f.close()

