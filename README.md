# Diabetes Indicator Data Analysis - CS422

Diabetes is a disease that affects over 400 million people worldwide. It is characterized by the inability of your body to produce or utilize the hormone insulin to break down glucose. In other words, it affects how your body turns food into energy. Despite how commonplace the condition is and the amount of resources available, it frequently goes undiagnosed for years. 

The goal of this project is to accurately detect diabetes in an individual by applying and comparing the performance of various commonly used machine learning algorithms.


# Dataset
This analysis was performed using a dataset of 253,680 responses from a telephone survey conducted by the CDC in 2015. 

This dataset was originally comprised of 330 features, but has been reduced to 21 features for the purposes of this project.  The features that were included in this analysis ranged from relevant health conditions to lifestyle elements and is represented as a binary (0/1) response. The resultant variable consists of two classes: the absence of diabetes, and pre-diabetic or diabetic.


# Conclusion
We demonstrated 3 different machine learning methods (KNN, Logistic Regression, and Neural Network) to generate a predictive model to determine whether or not an individual has diabetes, depending on certain health indicators. The F-score, which is a measure for accuracy and accounts for both False - Positives and False - Negatives, was used to compare the models which yielded 0.86, 0.87, and 0.87 respectively. The ROC curve was another determining performance evaluator that was used to analyze the cut-off threshold for classification, which yielded 0.77, 0.82, and 0.83 respectively. 

As a result of this analysis, we were able to determine that the ANN model was the most accurate and effective method. 

