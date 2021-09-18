# Databricks notebook source
# MAGIC %md
# MAGIC # Practice Example

# COMMAND ----------

# Helper packages
library(dplyr)       # for data manipulation
library(ggplot2)     # for data visualization
library(stringr)     # for string functionality

# Modeling packages
library(cluster)     # for general clustering algorithms
##library(factoextra)  # for visualizing cluster results

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Grocery items and quantities purchased. Each observation represents a single basket of goods that were purchased together.
# MAGIC Problem type: unsupervised basket analysis
# MAGIC response variable: NA
# MAGIC features: 42
# MAGIC observations: 2,000
# MAGIC objective: use attributes of each basket to identify common groupings of items purchased together.
# MAGIC access: available on the companion website for this book

# COMMAND ----------

# URL to download/read in the data
url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"

# Access data
my_basket <- readr::read_csv(url)

# Print dimensions
dim(my_basket)
## [1] 2000   42

# Peek at response variable
my_basket

# COMMAND ----------

url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"
my_basket <- readr::read_csv(url)

# COMMAND ----------

head(my_basket)

# COMMAND ----------

# Initialize total within sum of squares error: wss
wss <- 0

# Look over 1 to 15 possible clusters
for (i in 1:25) {
  # Fit the model: km.out
  km.out <- kmeans(my_basket, centers = i, nstart = 20, iter.max = 50)
  # Save the within cluster sum of squares
  wss[i] <- km.out$tot.withinss
}

# Produce a scree plot
plot(1:25, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# COMMAND ----------


