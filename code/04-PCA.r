# Databricks notebook source
# MAGIC %md 
# MAGIC # Principal component analysis (PCA)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Dimensionality reduction techniques are widely used and versatile techniques that can be used to:
# MAGIC 
# MAGIC find structure in features
# MAGIC pre-processing for other ML algorithms, and
# MAGIC aid in visualisation.
# MAGIC The basic principle of dimensionality reduction techniques is to transform the data into a new space that summarise properties of the whole data set along a reduced number of dimensions. These are then ideal candidates used to visualise the data along these reduced number of informative dimensions.

# COMMAND ----------

# MAGIC %md
# MAGIC Principal Component Analysis (PCA) is a technique that transforms the original n-dimensional data into a new n-dimensional space.
# MAGIC 
# MAGIC These new dimensions are linear combinations of the original data, i.e. they are composed of proportions of the original variables.
# MAGIC Along these new dimensions, called principal components, the data expresses most of its variability along the first PC, then second, …
# MAGIC Principal components are orthogonal to each other, i.e. non-correlated.
# MAGIC Original data (left). PC1 will maximise the variability while minimising the residuals (centre). PC2 is orthogonal to PC1 (right).
# MAGIC Figure 4.8: Original data (left). PC1 will maximise the variability while minimising the residuals (centre). PC2 is orthogonal to PC1 (right).
# MAGIC 
# MAGIC In R, we can use the prcomp function.
# MAGIC 
# MAGIC Let’s explore PCA on the iris data. While it contains only 4 variables, is already becomes difficult to visualise the 3 groups along all these dimensions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## In other words
# MAGIC #### 2 main goals of dimensionality reduciton
# MAGIC - find structure in features
# MAGIC - aid in visualization
# MAGIC #### PCA has 3 goals
# MAGIC - find a linear combintion of variables to create principle components
# MAGIC - maintain as much variance in the data as possible
# MAGIC - principal components are uncorrelated (i.e. orthogonoal to each other)
# MAGIC #### intuition
# MAGIC - with an x y correlation scatter plot, the best 1 dimension to explain the variance in the data is the linear regression line
# MAGIC - this is the first prinipal component
# MAGIC - then the distance of the points from the line is the component score (I don’t really understand this part, but I get how the line is simple way to explain the two dimensional data and explains most of the variation in the data.)
# MAGIC 
# MAGIC ##### Example: Principle components with iris dataset
# MAGIC 
# MAGIC - center and scale - for each point, subtract the mean and divide by the sd
# MAGIC - the summary of the model shows you the proportion of variance explained by each principal component
# MAGIC - I think the rotation is the distance of the point from each principal component or something like that.

# COMMAND ----------

set.seed(1)
xy <- data.frame(x = (x <- rnorm(50, 2, 1)),
                 y = x + rnorm(50, 1, 0.5))
pca <- prcomp(xy)
z <- cbind(x = c(-1, 1), y = c(0, 0))
zhat <- z %*% t(pca$rotation[, 1:2])
zhat <- scale(zhat, center = colMeans(xy), scale = FALSE)
par(mfrow = c(1, 3))
plot(xy, main = "Orignal data (2 dimensions)")
plot(xy, main = "Orignal data with PC1")
abline(lm(y ~ x, data = data.frame(zhat - 10)), lty = "dashed")
grid()
plot(pca$x, main = "Data in PCA space")
grid()

# COMMAND ----------

pairs(iris[, -5], col = iris[, 5], pch = 19)


# COMMAND ----------

# MAGIC %md
# MAGIC In R, we can use the `prcomp` function. 
# MAGIC 
# MAGIC Let's explore PCA on the `iris` data. While it contains only 4
# MAGIC variables, is already becomes difficult to visualise the 3 groups
# MAGIC along all these dimensions.

# COMMAND ----------

irispca <- prcomp(iris[, -5])
summary(irispca)

# COMMAND ----------

?prcomp


# COMMAND ----------

#A summary of the `prcomp` output shows that along PC1 along, we are able to retain over 92% of the total variability in the data.

## boxplot(irispca$x[, 1] ~ iris[, 5], ylab = "PC1")
hist(irispca$x[iris$Species == "setosa", 1],
     xlim = range(irispca$x[, 1]), col = "#FF000030",
     xlab = "PC1", main = "PC1 variance explained 92%")
rug(irispca$x[iris$Species == "setosa", 1], col = "red")
hist(irispca$x[iris$Species == "versicolor", 1], add = TRUE, col = "#00FF0030")
rug(irispca$x[iris$Species == "versicolor", 1], col = "green")
hist(irispca$x[iris$Species == "virginica", 1],  add = TRUE, col = "#0000FF30")
rug(irispca$x[iris$Species == "virginica", 1], col = "blue")

# COMMAND ----------

plot(irispca)

# COMMAND ----------

# MAGIC %md
# MAGIC A biplot features all original points re-mapped (rotated) along the first two PCs as well as the original features as vectors along the same PCs. Feature vectors that are in the same direction in PC space are also correlated in the original data space.

# COMMAND ----------

?biplot

# COMMAND ----------

biplot(irispca)


# COMMAND ----------

# MAGIC %md
# MAGIC One important piece of information when using PCA is the proportion of variance explained along the PCs, in particular when dealing with high dimensional data, as PC1 and PC2 (that are generally used for visualisation), might only account for an insufficient proportion of variance to be relevant on their own.
# MAGIC 
# MAGIC In the code chunk below, I extract the standard deviations from the PCA result to calculate the variances, then obtain the percentage of and cumulative variance along the PCs.

# COMMAND ----------

var <- irispca$sdev^2
(pve <- var/sum(var))

#Proportion of Variance Explained (PVE)

# COMMAND ----------

cumsum(pve)

# COMMAND ----------

# MAGIC %md
# MAGIC Challenge
# MAGIC 
# MAGIC - Repeat the PCA analysis on the iris dataset above, reproducing the biplot and preparing a barplot of the percentage of variance explained by each PC.
# MAGIC - It is often useful to produce custom figures using the data coordinates in PCA space, which can be accessed as x in the prcomp object. Reproduce the PCA plots below, along PC1 and PC2 and PC3 and PC4 respectively.

# COMMAND ----------

par(mfrow = c(1, 2))
plot(irispca$x[, 1:2], col = iris$Species)
plot(irispca$x[, 3:4], col = iris$Species)

# COMMAND ----------

par(mfrow = c(1, 2))
plot(irispca$x[, c(1,3)], col = iris$Species)
plot(irispca$x[, 2:3], col = iris$Species)

# COMMAND ----------

# MAGIC %md
# MAGIC #Data pre-processing
# MAGIC We haven’t looked at other prcomp parameters, other that the first one, x. There are two other ones that are or importance, in particular in the light of the section on pre-processing above, which are center and scale.. The former is set to TRUE by default, while the second one is set the FALSE.
# MAGIC 
# MAGIC Challenge
# MAGIC 
# MAGIC Repeat the analysis comparing the need for scaling on the mtcars dataset, but using PCA instead of hierarchical clustering. When comparing the two.

# COMMAND ----------

par(mfrow = c(1, 2))
biplot(prcomp(mtcars, scale = FALSE), main = "No scaling")  ## 1
biplot(prcomp(mtcars, scale = TRUE), main = "With scaling") ## 2


#Without scaling, disp and hp are the features with the highest loadings along PC1 and 2 (all others are negligible), which are also those with the highest units of measurement. Scaling removes this effect.

# COMMAND ----------

# MAGIC %md
# MAGIC # Final comments on PCA
# MAGIC Real datasets often come with missing values. In R, these should be encoded using NA. Unfortunately, PCA cannot deal with missing values, and observations containing NA values will be dropped automatically. This is a viable solution only when the proportion of missing values is low.
# MAGIC 
# MAGIC It is also possible to impute missing values. This is described in greater details in the Data pre-processing tutorials.
# MAGIC 
# MAGIC Finally, we should be careful when using categorical data in any of the unsupervised methods described above. Categories are generally represented as factors, which are encoded as integer levels, and might give the impression that a distance between levels is a relevant measure (which it is not, unless the factors are ordered). In such situations, categorical data can be dropped, or it is possible to encode categories as binary dummy variables. For example, if we have 3 categories, say A, B and C, we would create two dummy variables to encode the categories as:

# COMMAND ----------

dfr <- data.frame(x = c(1, 0, 0),
                  y = c(0, 1, 0))
rownames(dfr) <- LETTERS[1:3]
knitr::kable(dfr)

# COMMAND ----------

# MAGIC %md
# MAGIC To perform dimension reduction techniques in R, generally, the data should be prepared as follows:
# MAGIC 
# MAGIC - Data are in tidy format per Wickham and others (2014);
# MAGIC - Any missing values in the data must be removed or imputed;
# MAGIC - Typically, the data must all be numeric values (e.g., one-hot, label, ordinal encoding categorical features);
# MAGIC - Numeric data should be standardized (e.g., centered and scaled) to make features comparable.
