from xml.dom import minidom
from datetime import datetime
import mysql.connector

cnx = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='sof')
cursor = cnx.cursor()
cursor2 = cnx.cursor()
cursor3 = cnx.cursor()

i = 1
with open("Posts.xml") as posts:
    for line in posts:
        i += 1
        if i<=3 :
        # if i <= 8893589:
        #     print("skip "+str(i))
            continue
        # if line.strip() == "</posts>":
        #     break
        # print(line)
        xmldoc = minidom.parseString(line.strip())
        # print(xmldoc.documentElement.tagName)
        res = xmldoc.getElementsByTagName('row')
        print(str(i))

        if "Tags" not in res[0].attributes and "ParentId" in res[0].attributes:
            cursor2.execute(("select tags from posts where `Id` = %s"),(res[0].attributes['ParentId'].value,))
            tags = cursor2.fetchone()[0]
            update_post = ("update `posts` set Tags = %s where id = %s")
            data_post = (
                tags,
                res[0].attributes['Id'].value,
            )
            cursor.execute(update_post, data_post)
        # elif "ParentId" not in res[0].attributes and res[0].attributes['PostTypeId'].value == "2":
        #     print("Err in "+str(i))
        #     continue
        # elif "Tags" not in res[0].attributes and res[0].attributes['PostTypeId'].value == "1":
        #     print("Err in "+str(i))
        #     continue
        else:
            tags = res[0].attributes['Tags'].value.replace('><',' ').replace('<','').replace('>','') if "Tags" in res[0].attributes else None

        if tags != None:
            tg = tags.split(' ')
            for t in tg:
                add_tag = ("insert ignore into `tags` (p_id,tag) VALUES (%s, %s)")
                data_tag = (
                    res[0].attributes['Id'].value,
                    t,
                )
                cursor3.execute(add_tag, data_tag)
        else:
            print("Empty tag set")

        cnx.commit()
        # if i>10:
        #     break

cursor.close()
cursor2.close()
cursor3.close()
cnx.close()
