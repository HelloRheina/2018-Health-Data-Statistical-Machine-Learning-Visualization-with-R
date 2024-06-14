# 2018-Health-Data-Statistical-Machine-Learning-Visualization-with-R
R for analyzing data using ML 

Statistical machine learning develops models that learn from data and make predictions using statistical techniques. It This project explores health trends across US counties using the 2018 County Health Rankings Data.

## Goals:
Preprocess and clean the data.
Utilize R for data analysis.
Apply machine learning techniques to uncover patterns.
Evaluate model performance.
Interpret findings and draw conclusions.

## Tools and Data:
RStudio
R version 4.3.3
2018 County Health Rankings Data.xls

## Getting Started

Ensure the dataset is in the correct directory


### Overall technique

a. Linear regression 
A statistical technique for rank predictions, a linear modeling approach to quantify the relationship between a dependent variable and a set of independent variables. The principle of linear regression is to estimate the dependent variable's conditional probability distribution, given the independent variables' specific values. 

b. Logistic regression 
It is a method employed for classification tasks. The objective is to predict the likelihood of a data point belonging to a specific class. It uses statistical analysis to model the relationship between a binary dependent variable and one or more independent variables. This approach allows logistic regression to estimate the probability of an instance falling into a particular class. 

c. K-Nearest Neighbors (KNN)  
K-Nearest Neighbors (K-NN) can handle various data types. It makes predictions for new data points by finding similar existing data points (neighbors) and basing the prediction on those neighbors' characteristics. This allows K-NN to adapt to different patterns in the data without strict assumptions about the data's underlying structure. 

d. Decision tree  
A decision tree mimics a decision-making process like a flowchart. It starts with a question about the data, and then splits it into branches based on the answer. Each branch leads to another question or a final answer. This structure allows the tree to learn and predict by asking a series of questions about the data. 

e. K-means clustering 
K-means clustering is a technique for grouping similar data points. It begins by randomly selecting a set of K central points within the data space, called centroids. Each data point is then assigned to the closest centroid based on distance. Once all points are assigned, the centroids are recalculated to represent the center of their respective assigned points. This process of assigning points and recalculating centroids iterates until the centroids no longer significantly change, indicating that a stable grouping of the data has been achieved. This method assumes the number of clusters (K) is predetermined, and the goal is to assign each data point to one of these K groups.

f. Hierarchical clustering 
Hierarchical clustering is a method for grouping data points into a hierarchy of clusters. It connects data points that are close together (based on a distance measure) into small groups. These small groups can then be further connected into larger clusters, forming a hierarchy of groups that reflect varying levels of similarity within the data. 

## Acknowledgments

* reference from geeksforegeeks.com
