<h1> Fraud-Detection </h1>
<h2> Description </h2>
This project aims at detecting fraud in credit card transactions. The data and and problem statement is taken from Kaggle. 
https://www.kaggle.com/mlg-ulb/creditcardfraud

<h2>Algorithm Used </h2>
The dataset is highly skewed and any machine learning model that randomly predicts a transaction to be non-fraudulent would give 99% accuracy. 

Hence, common machine learning classification algorithms such as logistic regression would not work. We used multivariate gaussian algorithm to detect the anomalies in the data. The dataset has 30 independent variables and 1 dependent variable.

<h2> Usage </h2>
The code can be used for any data anomaly detection. You needs to modify the dimensions of dataset as per your dataset and use an appropriate value of threshold probability below which signifies the anomaly behaviour.
<h2> References </h2>
https://www.coursera.org/lecture/machine-learning/multivariate-gaussian-distribution-Cf8DF
