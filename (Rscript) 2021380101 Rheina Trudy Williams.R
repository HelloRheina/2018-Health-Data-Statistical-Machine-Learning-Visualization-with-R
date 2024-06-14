library(readxl)
library(ggplot2)
library(nnet)
library(class)
library(rpart)
library(dplyr)
library(reshape2)
library(viridis)
library(rpart.plot)
library(dynamicTreeCut)
library(dendextend)

# Load the data
X2018_health <- read_excel("~/UNI/SEMESTER 6/Statistical Machine Learning/2018_health.xls", sheet = "Outcomes & Factors Rankings")
#the data souce needs to be replaced according to the user's library
health_data <- X2018_health
sum(is.na(health_data))
str(health_data)
colnames(health_data) <- c("FIPS", "State", "County", "NumRankedCounties", "HealthOutcomesRank", "HealthOutcomesQuartile", "HealthFactorsRank", "HealthFactorsQuartile")
str(health_data)
# preparation
health_data$State <- as.factor(health_data$State)
health_data$County <- as.factor(health_data$County)
health_data$HealthOutcomesQuartile <- as.factor(health_data$HealthOutcomesQuartile)
health_data$HealthFactorsQuartile <- as.factor(health_data$HealthFactorsQuartile)
health_data$HealthOutcomesRank <- as.numeric(health_data$HealthOutcomesRank)
health_data$HealthFactorsRank <- as.numeric(health_data$HealthFactorsRank)
str(health_data)
# Remove rows with missing values in specific columns
complete_rows <- complete.cases(health_data[, c("HealthOutcomesRank", "HealthFactorsRank")])
health_data <- health_data[complete_rows, ]
sum(is.na(health_data))

# 1. Linear Regression
# model
lm_model <- lm(HealthOutcomesRank ~ HealthFactorsRank, data = health_data)
summary(lm_model)
# Plot : Health Outcomes Rank vs. Health Factors Rank
ggplot(health_data, aes(x = HealthFactorsRank, y = HealthOutcomesRank)) +
  geom_point(aes(color = State), size = 2) +
  stat_smooth(method = "lm", color = "red") +
  labs(title = "Health Outcomes Rank vs. Health Factors Rank",
       x = "Health Factors Rank",
       y = "Health Outcomes Rank") +
  theme_classic()
# Sample predictions for new data: rank
new_data_lmpredict <- data.frame(HealthFactorsRank = c(20, 45, 70))
predicted_outcomes <- predict(lm_model, new_data_lmpredict)
predicted_outcomes
# Plot : residuals vs fitted values
ggplot(health_data, aes(x = fitted(lm_model), y = residuals(lm_model))) +
  geom_point(alpha = 0.3) +
  labs(title = "Residuals vs Fitted Values",
       x = "Fitted Values",
       y = "Residuals") +
  theme_classic()
# Model
lm_model_interaction <- lm(HealthOutcomesRank ~ HealthFactorsRank * HealthOutcomesQuartile, data = health_data)
summary(lm_model_interaction)
predicted_outcomes <- predict(lm_model_interaction, newdata = health_data)
# Plot: prediction outcomes
ggplot(health_data, aes(x = HealthFactorsRank, y = HealthOutcomesRank, color = HealthOutcomesQuartile)) +
  geom_point(alpha = 0.3) +
  geom_line(aes(y = predicted_outcomes), color = "black", linetype = "dashed", size = 1) +
  labs(title = "Health Outcomes Rank vs. Health Factors Rank (by Quartile)",
       x = "Health Factors Rank",
       y = "Health Outcomes Rank",
       color = "Health Outcomes Quartile") +
  theme_classic()
# Preparation for next model
filtered_data <- health_data %>%
  filter(HealthOutcomesQuartile != "NR" & HealthOutcomesQuartile != "Quartile") %>%
  droplevels() %>%
  as.data.frame()

# 2. Logistic Regression
# Multinomial logistic regression model
multinom_model <- multinom(HealthOutcomesQuartile ~ HealthFactorsRank, data = filtered_data)
summary(multinom_model)
# Predictions
health_factors_range <- seq(min(filtered_data$HealthFactorsRank), max(filtered_data$HealthFactorsRank), length = 100)
predicted_probs <- predict(multinom_model, newdata = data.frame(HealthFactorsRank = health_factors_range), type = "probs")
# Plot:predicted probabilities
ggplot(data.frame(HealthFactorsRank = health_factors_range, predicted_probs)) +
  geom_line(aes(x = HealthFactorsRank, y = predicted_probs[, 1], color = "Quartile 1"), linetype = "dashed") +
  geom_line(aes(x = HealthFactorsRank, y = predicted_probs[, 2], color = "Quartile 2"), linetype = "dashed") +
  geom_line(aes(x = HealthFactorsRank, y = predicted_probs[, 3], color = "Quartile 3"), linetype = "dashed") +
  geom_line(aes(x = HealthFactorsRank, y = predicted_probs[, 4], color = "Quartile 4"), linetype = "dashed") +
  geom_line(aes(x = HealthFactorsRank, y = predicted_probs[, 5], color = "Quartile 5"), linetype = "dashed") +
  labs(title = "Predicted Probability of Health Outcome Quartiles by Health Factors Rank",
       x = "Health Factors Rank",
       y = "Predicted Probability",
       color = "Health Outcome Quartile") +
  scale_color_manual(values = c("pink", "red", "green", "blue", "brown")) +
  theme_classic()

# 3. KNN
# Normalize data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
health_data_norm <- as.data.frame(lapply(health_data[, c("HealthOutcomesRank", "HealthFactorsRank")], normalize))
# Preparation for KNN
set.seed(123)
train_index <- sample(1:nrow(health_data_norm), 0.7 * nrow(health_data_norm))
train_data <- health_data_norm[train_index, ]
test_data <- health_data_norm[-train_index, ]
# KNN model
knn_pred <- knn(train_data, test_data, cl = health_data$HealthOutcomesQuartile[train_index], k = 3)
confusion_matrix <- table(knn_pred, health_data$HealthOutcomesQuartile[-train_index])
summary(knn_pred)
table(knn_pred, health_data$HealthOutcomesQuartile[-train_index])
# Plot: confusion matrix
melted_confusion <- melt(confusion_matrix)
names(melted_confusion) <- c("Predicted_or_Actual", "Actual_Quartile", "Count")
ggplot(melted_confusion, aes(x = Actual_Quartile, y = Predicted_or_Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), vjust = 1, color = "black") +
  scale_fill_viridis_c() +
  labs(title = "KNN Model Confusion Matrix",
       x = "Actual Health Outcome Quartile",
       y = "Predicted Quartile",
       fill = "Number of Data Points") +
  theme_minimal()
# Plot: predicted classes
test_data$predicted_class <- knn_pred
ggplot(test_data, aes(x = HealthOutcomesRank, y = HealthFactorsRank, color = factor(predicted_class))) +
  geom_point() +
  labs(title = "KNN Predicted Classes",
       x = "Health Outcomes Rank (Normalized)",
       y = "Health Factors Rank (Normalized)",
       color = "Predicted Health Outcome Quartile")
# Plot: decision boundary
x <- seq(min(health_data_norm$HealthOutcomesRank), max(health_data_norm$HealthOutcomesRank), length.out = 100)
y <- seq(min(health_data_norm$HealthFactorsRank), max(health_data_norm$HealthFactorsRank), length.out = 100)
grid <- expand.grid(HealthOutcomesRank = x, HealthFactorsRank = y)
grid$knn_pred <- knn(train_data, grid, cl = health_data$HealthOutcomesQuartile[train_index], k = 3)
ggplot(grid, aes(x = HealthOutcomesRank, y = HealthFactorsRank, color = factor(knn_pred))) +
  geom_point() +
  scale_color_manual(values = c("1" = "blue", "2" = "green", "3" = "red", "4" = "orange", "5" = "purple")) +
  labs(title = "Decision Boundary of KNN Classifier",
       x = "Health Outcomes Rank (Normalized)",
       y = "Health Factors Rank (Normalized)",
       color = "Predicted Health Outcome Quartile") +
  theme_minimal() +
  geom_point(data = test_data, aes(color = factor(health_data$HealthOutcomesQuartile[-train_index])), size = 3, alpha = 0.5)

# 4. Decision Tree
# Decision tree model
tree_model <- rpart(HealthOutcomesQuartile ~ HealthFactorsRank, data = health_data, method = "class")
print(tree_model)
summary(tree_model)
rpart.plot(tree_model)

# 5. K-means Clustering
# K-means clustering
kmeans_model <- kmeans(health_data_norm, centers = 4)
health_data$kmeans_cluster <- as.factor(kmeans_model$cluster)
table(health_data$kmeans_cluster)
summary(kmeans_model)
# Plot
data_kmeans <- health_data_norm[, c("HealthOutcomesRank", "HealthFactorsRank")]
data_kmeans$cluster <- as.factor(kmeans_model$cluster)
ggplot(data_kmeans, aes(x = HealthOutcomesRank, y = HealthFactorsRank, color = cluster)) +
  geom_point() +
  stat_ellipse(aes(group = cluster), type = "t", linetype = "solid", alpha = 0.5) +
  labs(title = "K-Means Clustering Results",
       x = "Health Outcomes Rank (Normalized)",
       y = "Health Factors Rank (Normalized)",
       color = "Cluster") +
  scale_color_manual(values = c("#2E9FDF", "#00AFBB", "#E7B800", "#FF7F0E")) +
  theme_classic()

# 6. Hierarchical Clustering
# Model
dist_matrix <- dist(health_data_norm, method = "euclidean")
hclust_model <- hclust(dist_matrix, method = "ward.D2")
summary(hclust_model)
print(hclust_model)

# Plot : dendrogram
plot(hclust_model)
dynamic_clusters <- cutreeDynamic(hclust_model, distM = as.matrix(dist_matrix))
plot(hclust_model, labels = FALSE, main = "Optimized Cluster Dendrogram", xlab = "", sub = "", cex = 0.6)
rect.hclust(hclust_model, k = max(dynamic_clusters), border = "red")
dend <- as.dendrogram(hclust_model)
dend <- color_branches(dend, k = max(dynamic_clusters))
dend <- color_labels(dend, k = max(dynamic_clusters))
plot(dend, main = "Optimized Cluster Dendrogram")

