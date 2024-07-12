Titanic Survival Prediction using Logistic Regression
Project Overview
This project involves predicting the survival of passengers on the Titanic using a Logistic Regression model. The dataset is a well-known dataset from the Titanic disaster, provided by sklearn.

Dataset Description
The dataset consists of information about passengers such as age, gender, class, etc. The task is to use this information to predict whether a passenger survived the Titanic disaster.

Files
train.csv: Training dataset with features (X_train) and target (Y_train).
test.csv: Test dataset with features (X_test).
Model
A Logistic Regression model is used for this binary classification task. The model predicts whether a passenger survived (1) or did not survive (0).

Getting Started
Prerequisites
Make sure you have the following libraries installed:

pandas
numpy
scikit-learn
Project Structure: -
├── data
│   ├── train.csv
│   ├── test.csv
├── src
│   ├── titanic_logistic_regression.py
├── README.md


Results
The predictions for the test dataset will be saved in a file named titanic_predict.csv in the root directory.
