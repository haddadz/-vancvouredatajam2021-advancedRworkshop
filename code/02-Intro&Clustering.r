# Databricks notebook source
# MAGIC %md
# MAGIC #Introduction & k-means clustering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In unsupervised learning, no labels are provided, and the learning algorithm focuses solely on detecting structure in unlabelled input data. One generally differentiates between
# MAGIC 
# MAGIC - Clustering, where the goal is to find homogeneous subgroups within the data; the grouping is based on distance between observations.
# MAGIC 
# MAGIC - Dimensionality reduction, where the goal is to identify patterns in the features of the data. Dimensionality reduction is often used to facilitate visualisation of the data, as well as a pre-processing method before supervised learning.
# MAGIC 
# MAGIC Unsupservied learning presents specific challenges and benefits:
# MAGIC 
# MAGIC - there is no single goal in Unsupervised learning
# MAGIC - there is generally much more unlabelled data available than labelled data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## k-means clustering

# COMMAND ----------

# MAGIC %md
# MAGIC The k-means clustering algorithms aims at partitioning n observations into a fixed number of k clusters. The algorithm will find homogeneous clusters.
# MAGIC 
# MAGIC In R, we use
# MAGIC 
# MAGIC `stats::kmeans(x, centers = 3, nstart = 10)`

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Let's get started

# COMMAND ----------

##prepare packages

#install.packages(“tidyverse”)       # for data work & visualization
#install.packages(“cluster”)         # for cluster modeling 
#install.packages("reshape2")        # for melting data
# note : not required if already installed
library(tidyverse)
library(cluster)
library(reshape2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load our Dataset

# COMMAND ----------

#load the iris dataset 
data(iris)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore our Data Set (EDA)

# COMMAND ----------

head(iris)

# COMMAND ----------

glimpse(iris)
#head(iris)
#View(iris)

# COMMAND ----------

#Visualize the data 
## Sepal-Length vs Sepal-Wdith

ggplot(iris)+
 geom_point(aes(x = Sepal.Length, y = Sepal.Width), stroke = 2)+
 facet_wrap(~ Species)+ 
 labs(x = 'Sepal Length', y = 'Sepal Width')#+
 #theme_bw()

# COMMAND ----------

#Petal-Length vs. Petal-Width
ggplot(iris)+
 geom_point(aes(x = Petal.Length, y = Petal.Width), stroke = 2)+
 facet_wrap(~ Species)+ 
 labs(x = 'Petal Length', y = 'Petal Width')

# COMMAND ----------

#Sepal-Length vs. Petal-Length
ggplot(iris)+
 geom_point(aes(x = Sepal.Length, y = Petal.Length), stroke = 2)+
 facet_wrap(~ Species)+ 
 labs(x = 'Sepal Length', y = 'Petal Length') #+theme_bw()

# COMMAND ----------

#Sepal-Width vs. Pedal-Width
ggplot(iris)+
 geom_point(aes(x = Sepal.Width, y = Petal.Width), stroke = 2)+
 facet_wrap(~ Species)+ 
 labs(x = 'Sepal Width', y = 'Pedal Width')+
 theme_bw()

# COMMAND ----------

#Box plots
ggplot(iris)+
 geom_boxplot(aes(x = Species, y = Sepal.Length, fill = Species))


# COMMAND ----------

ggplot(iris)+
 geom_boxplot(aes(x = Species, y = Sepal.Width, fill = Species))


# COMMAND ----------

ggplot(iris)+
 geom_boxplot(aes(x = Species, y = Petal.Length, fill = Species))


# COMMAND ----------

ggplot(iris)+
 geom_boxplot(aes(x = Species, y = Petal.Width, fill = Species))

# COMMAND ----------

# MAGIC %md
# MAGIC k-means clustering algorithms aims at partioning n observations into a fixed number of k clusters. 
# MAGIC Finding the homogeneous clusters. 
# MAGIC 
# MAGIC We use 
# MAGIC stats::kmeans(x, centers = 3, nstart = 10)
# MAGIC 
# MAGIC - x numeric data matrix
# MAGIC - centers is pre-defined # of clusters
# MAGIC - k-means has a random component which can be repeated nstart times to improve the returned model

# COMMAND ----------

?kmeans


# COMMAND ----------

# MAGIC %md
# MAGIC ### Example
# MAGIC 
# MAGIC - To learn about k-means, let’s use the iris dataset with the sepal and petal length variables only (to facilitate visualisation). Create such a data matrix and name it x
# MAGIC - Run the k-means algorithm on the newly generated data x, save the results in a new variable cl, and explore its output when printed.
# MAGIC - The actual results of the algorithms, i.e. the cluster membership can be accessed in the clusters element of the clustering result output. Use it to colour the inferred clusters to generate a figure like that shown below.

# COMMAND ----------

i <- grep("Length", names(iris))
x <- iris[, i]
cl <- kmeans(x, 3, nstart = 10)
plot(x, col = cl$cluster)

# COMMAND ----------

stats::kmeans(x, centers = 3, nstart = 10)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### How does k-means work
# MAGIC - Initialisation: randomly assign class membership

# COMMAND ----------

set.seed(12)
init <- sample(3, nrow(x), replace = TRUE)
plot(x, col = init)

# COMMAND ----------

# MAGIC %md
# MAGIC k-means random intialisation
# MAGIC 
# MAGIC Iteration:
# MAGIC 
# MAGIC - Calculate the centre of each subgroup as the average position of all observations is that subgroup.
# MAGIC - Each observation is then assigned to the group of its nearest centre.     
# MAGIC It’s also possible to stop the algorithm after a certain number of iterations, or once the centres move less than a certain distance.

# COMMAND ----------

par(mfrow = c(1, 2))
plot(x, col = init)
centres <- sapply(1:3, function(i) colMeans(x[init == i, ], ))
centres <- t(centres)
points(centres[, 1], centres[, 2], pch = 19, col = 1:3)

tmp <- dist(rbind(centres, x))
tmp <- as.matrix(tmp)[, 1:3]

ki <- apply(tmp, 1, which.min)
ki <- ki[-(1:3)]

plot(x, col = ki)
points(centres[, 1], centres[, 2], pch = 19, col = 1:3)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Termination: Repeat iteration until no point changes its cluster membership.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Ex](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif?psid=1&width=320&height=320) 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC k-means convergence (credit Wikipedia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Due to the random initialisation, one can obtain different clustering results. When k-means is run multiple times, the best outcome, i.e. the one that generates the smallest total within cluster sum of squares (SS), is selected. The total within SS is calculated as:
# MAGIC 
# MAGIC For each cluster results:
# MAGIC 
# MAGIC - for each observation, determine the squared euclidean distance from observation to centre of cluster
# MAGIC - sum all distances
# MAGIC Note that this is a local minimum; there is no guarantee to obtain a global minimum.
# MAGIC 
# MAGIC Challenge:
# MAGIC 
# MAGIC - Repeat k-means on our x data multiple times, setting the number of iterations to 1 or greater and check whether you repeatedly obtain the same results. Try the same with random data of identical dimensions.

# COMMAND ----------

cl1 <- kmeans(x, centers = 3, nstart = 10)
cl2 <- kmeans(x, centers = 3, nstart = 10)
table(cl1$cluster, cl2$cluster)

# COMMAND ----------

cl1 <- kmeans(x, centers = 3, nstart = 1)
cl2 <- kmeans(x, centers = 3, nstart = 1)
table(cl1$cluster, cl2$cluster)

# COMMAND ----------

set.seed(42)
xr <- matrix(rnorm(prod(dim(x))), ncol = ncol(x))
cl1 <- kmeans(xr, centers = 3, nstart = 1)
cl2 <- kmeans(xr, centers = 3, nstart = 1)
table(cl1$cluster, cl2$cluster)

# COMMAND ----------

diffres <- cl1$cluster != cl2$cluster
par(mfrow = c(1, 2))
plot(xr, col = cl1$cluster, pch = ifelse(diffres, 19, 1))
plot(xr, col = cl2$cluster, pch = ifelse(diffres, 19, 1))


#Different k-means results on the same (random) data

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to determine the number of clusters
# MAGIC - Run k-means with k=1, k=2, …, k=n
# MAGIC - Record total within SS for each value of k.
# MAGIC - Choose k at the elbow position, as illustrated below.

# COMMAND ----------

# Initialize total within sum of squares error: wss
wss <- 0

# For 1 to 15 cluster centers
for (i in 1:15) {
  km.out <- kmeans(x, centers = i, nstart = 20)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}

# Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# COMMAND ----------

# MAGIC %md
# MAGIC ## In other words
# MAGIC ### k-means Clustering
# MAGIC k-means clustering is a method of vector quantization, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.
# MAGIC Find the optimal number of clusters by Elbow Method

# COMMAND ----------

ks <- 1:5
tot_within_ss <- sapply(ks, function(k) {
    cl <- kmeans(x, k, nstart = 10)
    cl$tot.withinss
})
plot(ks, tot_within_ss, type = "b")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### How kmeans() works and practical matters
# MAGIC ##### Process of k-means:
# MAGIC 
# MAGIC - randomly assign all points to a cluster
# MAGIC - calculate center of each cluster
# MAGIC - convert points to cluster of nearest center
# MAGIC - if no points changed, done, otherwise repeat
# MAGIC - calculate new center based new points
# MAGIC - convert points to cluster of nearest center
# MAGIC - and so on
# MAGIC ##### model selection:
# MAGIC 
# MAGIC - best outcome is based on total within cluster sum of squares
# MAGIC - run many times to get global optimum
# MAGIC - R will automaitcally take the run with the lowest total withinss
# MAGIC determining number of clusters
# MAGIC 
# MAGIC ##### scree plot
# MAGIC - look for the elbow
# MAGIC - find where addition on new cluster does not change best withinss much
# MAGIC - there ususally is no clear elbow in real world data

# COMMAND ----------

##the elbow point : k(centers) = 3
##Apply kmeans function to the feature columns
set.seed(123)
km <- kmeans( x = iris[, -5] , centers = 3)
yclus <- km$cluster
table(yclus)

# COMMAND ----------

#the kmeans has grouped the data into three clusters- 1, 2 & 3 having 50, 62 & 38 observations respectively.
#Visualize the kmeans clusters

clusplot(iris[, -5],
 yclus,
 lines = 0,
 shade = TRUE,
 color = TRUE,
 labels = 0,
 plotchar = FALSE,
 span = TRUE,
 main = paste('Clusters of Iris Flowers')
)

# COMMAND ----------

#Compare the clusters
iris$cluster.kmean <- yclus
cm <- table(iris$Species, iris$cluster.kmean)
cm

# COMMAND ----------

(50 + 48 + 36)/150 *100

#of the k-means cluster output matched with the actual Species clusters. versicolor(Cluster 2) & virginica(Cluster 3) have some overlapping features which is also apparent from the cluster visualizations.

# COMMAND ----------


