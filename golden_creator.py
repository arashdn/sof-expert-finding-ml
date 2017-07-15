import mysql.connector

acceptance_ratio = 0.4
number_of_acc_answers = 10

cnx = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='sof')
cnx2 = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='sof')

cursor = cnx.cursor()

all_tags = []

query = ("SELECT c,tag FROM `php_tag_full`")
cursor.execute(query)
for (c, tag) in cursor:
    all_tags.append(tag)
cursor.close()


for tag in all_tags:
    print(tag)
    cursor = cnx.cursor()
    q = "select UserID,count(*) as c from php_user_answer where tag = '"+tag+"' and accepted = 1 and UserID is not null group by userID HAVING c >= "+str(number_of_acc_answers)+" order by c desc"
    cursor.execute(q)

    unaccepted_count = {}
    cursor2 = cnx2.cursor()
    query = ("select userId,count(aid) as c from php_user_answer where accepted = 0 and tag='"+tag+"' and userId is not null group by userid")
    cursor2.execute(query)
    for (UserID, c) in cursor2:
        unaccepted_count[UserID] = c
    cursor2.close()

    lst = []
    for (UserID, c) in cursor:
        cnt = unaccepted_count[UserID]+c if UserID in unaccepted_count else c
        if c/cnt > acceptance_ratio:
            lst.append(UserID)
    cursor.close()
    f = open("golden/DataSetFor"+tag+".csv", "w")
    print("UserId,Tag",file=f)
    for l in lst:
        print(str(l)+","+tag, file=f)
    f.close()
