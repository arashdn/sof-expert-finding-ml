import mysql.connector
main_tag = 'java'
cnx = mysql.connector.connect(user='root', password='root', host='127.0.0.1', database='sof')
cursor = cnx.cursor()


tag_cnt = {}
all_tags = []
cursor2 = cnx.cursor()
query = ("SELECT c,tag FROM `"+main_tag+"_tag_full`")
cursor.execute(query)
for (c, tag) in cursor:
    tag_cnt[tag] = c
    all_tags.append(tag)
cursor2.close()

ln = len(all_tags)

intersect = {}
for t in all_tags:
    intersect[t] = {}
    for t2 in all_tags:
        intersect[t][t2] = 0;

# print(dist)

query = ("select id,tags from posts where id in (select p_id from "+main_tag+"_qa where posttypeid=1)")

cursor.execute(query)
print("Query Done!")

cnt = 0;
for (id, tags) in cursor:
    cnt += 1
    if tags is None or tags == "":
        print("Empty tag set")
        continue

    print(str(cnt))
    tg = tags.split(' ')
    for i in range(ln):
        j = i + 1
        while j < ln:
            if all_tags[i] in tg and all_tags[j] in tg:
                intersect[all_tags[i]][all_tags[j]] += 1
                intersect[all_tags[j]][all_tags[i]] += 1
            j += 1
    # if cnt > 50000:
    #     break

f = open("./data/dists_"+main_tag+".csv","w")
s = "," # first empty cell
for i in range(ln):
    s += all_tags[i]+","
s = s[:-1]
print(s, file=f)
for i in range(ln):
    s = all_tags[i]
    for j in range(ln):
        num = 1
        if i != j:
            num = float(intersect[all_tags[i]][all_tags[j]]) / float(tag_cnt[all_tags[i]] + tag_cnt[all_tags[j]] - intersect[all_tags[i]][all_tags[j]])
        s += ","+('{0:f}'.format(1 - num))

    print(s, file=f)
f.close()
cursor.close()
cnx.close()
