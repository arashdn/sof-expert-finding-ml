--- show engine innodb status \G;

---- get Top 100 frequent tags:
select count(*) as c,tag from tags group by tag order by c desc limit 100;
--run time: 15min



---- export php posts
create table php_qa as
select id,PostTypeId from posts where id in (select p_id from tags where tag='php');
-- Query OK, 2012583 rows affected (12 min 37.51 sec)


--- php question and tag
create table php_q_tag as
select * from tags where p_id in (select p_id from php_qa where PostTypeId = 1);
-- Query OK, 2209964 rows affected (2 min 9.78 sec)


--- php answer and tag
create table php_a_tag as
select * from tags where p_id in (select p_id from php_qa where PostTypeId = 2);
-- Query OK, 3893520 rows affected (3 min 21.68 sec)


---- get Top 100 frequent php tags:
create table php_tag as 
select count(*) as c,tag from tags where p_id in(select p_id from php_qa where PostTypeId = 1) group by tag order by c desc limit 101;
-- Query OK, 100 rows affected (1 min 20.45 sec)
delete from php_tag where tag = 'php';
------
create table php_tag_full as 
select count(*) as c,tag from tags where p_id in(select p_id from php_qa where PostTypeId = 1) group by tag order by c desc limit 100;



---- distance
-- select count(p_id) from php_qa where posttypeid = 1 and p_id in (select p_id from tags where tag='mysql') and p_id in(select p_id from tags where tag='sql');

-- select count(p_id) from php_qa where posttypeid = 1 and EXISTS (select p_id from tags where tags.p_id = php_qa.p_id and tag='mysql') and EXISTS (select p_id from tags where tags.p_id = php_qa.p_id and tag='sql');



---- export java posts
create table java_qa as
select id,PostTypeId from posts where id in (select p_id from tags where tag='java');
-- 
-- Query OK, 2320883 rows affected (13 min 4.86 sec)

--- java question and tag
create table java_q_tag as
select * from tags where p_id in (select p_id from java_qa where PostTypeId = 1);
-- Query OK, 2667299 rows affected (2 min 13.13 sec)

---- get Top 100 frequent java tags:
create table java_tag as 
select count(*) as c,tag from tags where p_id in(select p_id from java_qa where PostTypeId = 1) group by tag order by c desc limit 101;
-- Query OK, 101 rows affected (1 min 28.63 sec)
delete from java_tag where tag = 'java';
------
create table java_tag_full as
select count(*) as c,tag from tags where p_id in(select p_id from java_qa where PostTypeId = 1) group by tag order by c desc, tag DESC limit 100;



---- export .net posts
create table dotnet_qa as
select id,PostTypeId from posts where id in (select p_id from tags where tag='.net');
-- Query OK, 628121 rows affected (11 min 46.32 sec)

--- dotnet question and tag
create table dotnet_q_tag as
select * from tags where p_id in (select p_id from dotnet_qa where PostTypeId = 1);
-- Query OK, 761567 rows affected (58.79 sec)


---- get Top 100 frequent .net tags:
create table dotnet_tag as 
select count(*) as c,tag from tags where p_id in(select p_id from dotnet_qa where PostTypeId = 1) group by tag order by c desc limit 101;
-- Query OK, 101 rows affected (50.36 sec)
delete from dotnet_tag where tag = '.net';




---- golden and post answer for php
ALTER TABLE `posts` ADD INDEX(`ParentId`);

# get answer,user,accepted,tag -->php
create table php_user_answer as
select aid,OwnerUserId as UserID,tag,accepted from php_all_qa,tags,posts where aid=posts.id and aid = tags.p_id;


# get all php questions
create table php_all_qa as
select p1.id as QId,p2.id as AId,IF(p1.AcceptedAnswerId = p2.Id , 1 , 0 ) as accepted from posts p1  inner join posts p2 on p1.id = p2.ParentId
where p1.id in(select p_id from php_q_tag);


# get answer,user,accepted,tag -->php
create table php_user_answer as
select aid,OwnerUserId as UserID,tag,accepted from php_all_qa,tags,posts where aid=posts.id and aid = tags.p_id;






