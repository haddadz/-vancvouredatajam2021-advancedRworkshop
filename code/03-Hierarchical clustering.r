# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction to hierarchical clustering

# COMMAND ----------

# MAGIC %md
# MAGIC Typically used when the number of clusters in not known a head of time
# MAGIC Two approaches. bottum up and top down. we will focus on bottum up process
# MAGIC - assign each point to its own cluster
# MAGIC - joing the two closesest custers/points into a new cluster
# MAGIC - keep going until there is one cluster
# MAGIC - they way you calculate the distance between clusters is a paramater and will be covered later
# MAGIC we have to first calculate the euclidean distance between all points (makes a big matriz) using the dist() function
# MAGIC - this is passed into the hclust() function

# COMMAND ----------

data(iris)

# COMMAND ----------

d <- dist(iris[, 1:4])
hcl <- hclust(d)
hcl

# COMMAND ----------

?dist

# COMMAND ----------

?hclust


# COMMAND ----------

plot(hcl)

# COMMAND ----------

# MAGIC %md
# MAGIC  ## Defining clusters
# MAGIC After producing the hierarchical clustering result, we need to cut the tree (dendrogram) at a specific height to defined the clusters. For example, on our test dataset above, we could decide to cut it at a distance around 1.5, with would produce 2 clusters.

# COMMAND ----------

# MAGIC %md
# MAGIC In R we can us the cutree function to
# MAGIC 
# MAGIC - cut the tree at a specific height: cutree(hcl, h = 1.5)
# MAGIC - cut the tree to get a certain number of clusters: cutree(hcl, k = 2)
# MAGIC 
# MAGIC Example
# MAGIC 
# MAGIC - Cut the iris hierarchical clustering result at a height to obtain 3 clusters by setting h.
# MAGIC - Cut the iris hierarchical clustering result at a height to obtain 3 clusters by setting directly k, and verify that both provide the same results.

# COMMAND ----------

plot(hcl)
abline(h = 3.9, col = "red")

# COMMAND ----------

cutree(hcl, k = 3)


# COMMAND ----------

cutree(hcl, h = 3.9)

# COMMAND ----------

identical(cutree(hcl, k = 3), cutree(hcl, h = 3.9))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Challenge
# MAGIC 
# MAGIC Using the same value k = 3, verify if k-means and hierarchical clustering produce the same results on the iris data.
# MAGIC 
# MAGIC Which one, if any, is correct?

# COMMAND ----------

km <- kmeans(iris[, 1:4], centers = 3, nstart = 10)
hcl <- hclust(dist(iris[, 1:4]))
table(km$cluster, cutree(hcl, k = 3))

# COMMAND ----------

par(mfrow = c(1, 2))
plot(iris$Petal.Length, iris$Sepal.Length, col = km$cluster, main = "k-means")
plot(iris$Petal.Length, iris$Sepal.Length, col = cutree(hcl, k = 3), main = "Hierarchical clustering")

# COMMAND ----------

## Checking with the labels provided with the iris data
table(iris$Species, km$cluster)

# COMMAND ----------

table(iris$Species, cutree(hcl, k = 3))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-processing
# MAGIC Many of the machine learning methods that are regularly used are sensitive to difference scales. This applies to unsupervised methods as well as supervised methods, as we will see in the next chapter.
# MAGIC 
# MAGIC A typical way to pre-process the data prior to learning is to scale the data, or apply principal component analysis (next section). Scaling assures that all data columns have a mean of 0 and standard deviation of 1.
# MAGIC 
# MAGIC In R, scaling is done with the scale function.
# MAGIC 
# MAGIC Challenge
# MAGIC 
# MAGIC Using the mtcars data as an example, verify that the variables are of different scales, then scale the data. To observe the effect different scales, compare the hierarchical clusters obtained on the original and scaled data.

# COMMAND ----------

##
colMeans(mtcars)
##        mpg        cyl       disp         hp       drat         wt       qsec 
##  20.090625   6.187500 230.721875 146.687500   3.596563   3.217250  17.848750 
##         vs         am       gear       carb 
##   0.437500   0.406250   3.687500   2.812500

hcl1 <- hclust(dist(mtcars))
hcl2 <- hclust(dist(scale(mtcars)))
par(mfrow = c(1, 2))
plot(hcl1, main = "original data")
plot(hcl2, main = "scaled data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional info
# MAGIC 
# MAGIC Clustering linkage and practical matters
# MAGIC 4 methods to measure distance between clusters
# MAGIC complete: pairwise similarty between all observations in cluster 1 and 2, uses largest of similarities
# MAGIC single: same as above but uses the smallest of similarities
# MAGIC average: same as above but uses average of similarities
# MAGIC centroid: finds centroid of cluster 1 and 2, uses similarity between tow centroids
# MAGIC rule of thumb
# MAGIC complete and average produce more balanced treess and are more commonly used
# MAGIC single fuses observations in one at a time and produces more unblanced trees
# MAGIC centroid can create inversion where clusters are put below single values. its not used often
# MAGIC practical matters
# MAGIC data needs to be scaled so that features have the same mean and standard deviation
# MAGIC normalized features have a mean of zero and a sd of one
# MAGIC – Linkage methods
# MAGIC # Cluster using complete linkage: hclust.complete
# MAGIC hclust.complete <- hclust(dist(x), method = "complete")
# MAGIC 
# MAGIC # Cluster using average linkage: hclust.average
# MAGIC hclust.average <- hclust(dist(x), method = "average")
# MAGIC 
# MAGIC # Cluster using single linkage: hclust.single
# MAGIC hclust.single <- hclust(dist(x), method = "single")
# MAGIC 
# MAGIC # Plot dendrogram of hclust.complete
# MAGIC plot(hclust.complete, main = "Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to k-means , we measure the (dis)similarity of observations using distance measures (e.g., Euclidean distance, Manhattan distance, etc.); the Euclidean distance is most commonly the default. However, a fundamental question in hierarchical clustering is: How do we measure the dissimilarity between two clusters of observations? A number of different cluster agglomeration methods (i.e., linkage methods) have been developed to answer this question. The most common methods are:
# MAGIC 
# MAGIC - Maximum or complete linkage clustering: Computes all pairwise dissimilarities between the elements in cluster 1 and the elements in cluster 2, and considers the largest value of these dissimilarities as the distance between the two clusters. It tends to produce more compact clusters.
# MAGIC - Minimum or single linkage clustering: Computes all pairwise dissimilarities between the elements in cluster 1 and the elements in cluster 2, and considers the smallest of these dissimilarities as a linkage criterion. It tends to produce long, “loose” clusters.
# MAGIC - Mean or average linkage clustering: Computes all pairwise dissimilarities between the elements in cluster 1 and the elements in cluster 2, and considers the average of these dissimilarities as the distance between the two clusters. Can vary in the compactness of the clusters it creates.
# MAGIC - Centroid linkage clustering: Computes the dissimilarity between the centroid for cluster 1 (a mean vector of length  
# MAGIC p
# MAGIC  , one element for each variable) and the centroid for cluster 2.
# MAGIC - Ward’s minimum variance method: Minimizes the total within-cluster variance. At each step the pair of clusters with the smallest between-cluster distance are merged. Tends to produce more compact clusters.
