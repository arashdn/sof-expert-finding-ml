from xml.dom import minidom
from datetime import datetime
import mysql.connector

cnx = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='sof')
cursor = cnx.cursor()

i = 1
with open("Posts.xml") as posts:
    for line in posts:
        i += 1
        if i<=3 :
        # if i< 22367535 :
        #     print("skip " +str(i))
            continue
        if line.strip() == "</posts>":
            break
        print(str(i))
        # print(line)
        xmldoc = minidom.parseString(line.strip())
        # print(xmldoc.documentElement.tagName)
        res = xmldoc.getElementsByTagName('row')

        tags = res[0].attributes['Tags'].value.replace('><',' ').replace('<','').replace('>','') if "Tags" in res[0].attributes else None

        add_post = ("INSERT INTO `posts` "+
                    "(`Id`, `PostTypeId`, `AcceptedAnswerId`, `ParentId`, `CreationDate`, `Score`, `ViewCount`, `Body`, `OwnerUserId`, "+
                    "`LastEditorUserId`, `LastEditorDisplayName`, `LastEditDate`, `LastActivityDate`, `Title`, `Tags`, `AnswerCount`, `CommentCount`, `FavoriteCount`, `CommunityOwnedDate`) "+
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

        data_post = (
            res[0].attributes['Id'].value if "Id" in res[0].attributes else None,
            res[0].attributes['PostTypeId'].value if "PostTypeId" in res[0].attributes else None,
            res[0].attributes['AcceptedAnswerId'].value if "AcceptedAnswerId" in res[0].attributes else None,
            res[0].attributes['ParentId'].value if "ParentId" in res[0].attributes else None,
            int(datetime.strptime(res[0].attributes['CreationDate'].value.replace('T', ' ').split('.')[0],'%Y-%m-%d %H:%M:%S').timestamp()) if "CreationDate" in res[0].attributes else None,
            res[0].attributes['Score'].value if "Score" in res[0].attributes else None,
            res[0].attributes['ViewCount'].value if "ViewCount" in res[0].attributes else None,
            res[0].attributes['Body'].value if "Body" in res[0].attributes else None,
            res[0].attributes['OwnerUserId'].value if "OwnerUserId" in res[0].attributes else None,
            res[0].attributes['LastEditorUserId'].value if "LastEditorUserId" in res[0].attributes else None,
            res[0].attributes['LastEditorDisplayName'].value if "LastEditorDisplayName" in res[0].attributes else None,
            int(datetime.strptime(res[0].attributes['LastEditDate'].value.replace('T', ' ').split('.')[0],'%Y-%m-%d %H:%M:%S').timestamp()) if "LastEditDate" in res[0].attributes else None,
            int(datetime.strptime(res[0].attributes['LastActivityDate'].value.replace('T', ' ').split('.')[0],'%Y-%m-%d %H:%M:%S').timestamp()) if "LastActivityDate" in res[0].attributes else None,
            res[0].attributes['Title'].value if "Title" in res[0].attributes else None,
            tags,
            res[0].attributes['AnswerCount'].value if "AnswerCount" in res[0].attributes else None,
            res[0].attributes['CommentCount'].value if "CommentCount" in res[0].attributes else None,
            res[0].attributes['FavoriteCount'].value if "FavoriteCount" in res[0].attributes else None,
            int(datetime.strptime(res[0].attributes['CommunityOwnedDate'].value.replace('T', ' ').split('.')[0],'%Y-%m-%d %H:%M:%S').timestamp()) if "CommunityOwnedDate" in res[0].attributes else None,
        )

        # Insert new employee
        cursor.execute(add_post, data_post)


        cnx.commit()
        # if i>10:
        #     break

cursor.close()
cnx.close()