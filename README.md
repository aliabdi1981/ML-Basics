# ML-Basics

## Import Required Libraries

import numpy as np
import pandas as pd

## Import the Data set as CSV File

DataSet = = pd.read_csv('.../dataset.csv')

## Define Main and Target sections for training and testing
 Assuming the data set is already clean and has no missing values or non-numerical values.

Training_part = DataSet.drop('testing_part_columns', axis=1)
Testing_part = DataSet['testing_part_columns']

## Import required SciKit-Learn Methods
In this case we use RandomForestClassifier and define it as ' clf ' 

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

## Import train-test-split method to split our data sets
Use this method to split the defined data sets in to 2 groups of Training and Testing.
The test_size defines the proportion of 80% for Training and 20% for Testing.

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(Training_part, Testing_part, test_size=0.2)

## Fit the model 
Fit the model to the RandomForestClassifier method as below.

clf.fit(X_train, y_train);

## Run the model
To run the model use the test parts of the data set.

clf.score(X_test, y_test)

## Export the model

- To export the model use the following code.
- Use desired name for the exported file.
- The file will be saved in the same directory as your project as a .pkl file

import pickle

pickle.dump(clf, open("sample_ml_model.pkl", "wb"))

## Import and run the Model

To import and run the model simply use the following code.

loaded_model = pickle.load(open("sample_ml_model.pkl", "rb"))
loaded_model.score(X_test, y_test)

## Notes:

- The returned value after running the model indicates the precision of the model which varies depending on the data set values. 
- Use your own desired names for data set and csv file according to your project files.

