# php
dist_php = read.csv("~/Desktop/PycharmProjects/expert_finding/cluster/data/dists_php.csv",header = TRUE, row.names = 1)
d_php = as.dist(dist_php)
cl_php = hclust(d_php)
res_php = cutree(cl_php, k = 32)
write.csv( res_php, file = "~/Desktop/PycharmProjects/expert_finding/cluster/data/res_php.csv")


# java
dist_java = read.csv("~/Desktop/PycharmProjects/expert_finding/cluster/data/dists_java.csv",header = TRUE, row.names = 1)
d_java = as.dist(dist_java)
cl_java = hclust(d_java)
res_java = cutree(cl_java, k = 22)
write.csv( res_java, file = "~/Desktop/PycharmProjects/expert_finding/cluster/data/res_java.csv")
