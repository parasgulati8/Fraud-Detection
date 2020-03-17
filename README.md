<h1> Fraud-Detection </h1>
<h2> Description </h2>
This project aims at detecting fraud in credit card transactions. The data and and problem statement is taken from Kaggle. 
https://www.kaggle.com/mlg-ulb/creditcardfraud

<h2>Algorithm Used </h2>
The dataset is highly skewed and any machine learning model that randomly predicts a transaction to be non-fraudulent would give 99% accuracy. 

Hence, common machine learning classification algorithms such as logistic regression would not work. We used multivariate gaussian algorithm to detect the anomalies in the data. 
The expression for univariate Gaussian is given by :

![univariate gaussian](https://latex.codecogs.com/gif.latex?%24%24p%28x%3B%5Cmu%2C%20%5Csigma%5E2%29%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%5Csigma%5E2%7D%7Dexp%5Cleft%28-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%28x-%5Cmu%29%5E2%5Cright%29%24%24)

For multivariate Gaussian, univariate gaussian probabilities for all the features are calculated and multiplied together. This product is a multivariate gaussian distribution and can be expresses as :
![Multivariate Gaussian](https://latex.codecogs.com/gif.latex?%24%24p%28x%3B%5Cmu%2C%20%5CSigma%29%3D%20%5Cfrac%7B1%7D%7B%7B%282%5Cpi%29%5E%7Bn/2%7D%5Cleft%20%7C%20%5CSigma%20%5Cright%20%7C%5E%7B1/2%7D%7D%7Dexp%5Cleft%28-%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu%29%5ET%5CSigma%5E%7B-1%7D%28x-%5Cmu%29%5Cright%29%24%24)

The dataset has 30 independent variables and 1 dependent variable.

<h2> Usage </h2>
The code can be used for any data anomaly detection. You needs to modify the dimensions of dataset as per your dataset and use an appropriate value of threshold probability below which signifies the anomaly behaviour.
<h2> References </h2>
https://www.coursera.org/lecture/machine-learning/multivariate-gaussian-distribution-Cf8DF
