import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pickle

# loading the csv data into panda frame
data = pd.read_csv('heart_disease_data.csv')

# print first 5 rows of the dataset
print("First Five Rows:")
print(data.head())

print("-------------------------------------")

# print the last 5 rows of the dataset
print("Last five Rows")
print(data.tail())

print("-------------------------------------")

# Rows and columns in the data
print("Shape:")
print(data.shape)

print("-------------------------------------")

# info about the data
print("Info:")
print(data.info())

print("-------------------------------------")

# check for missing values
print("Missing Values:")
print(data.isnull().sum())

print("-------------------------------------")

# data statistics
print("Data Statistics")
print(data.describe())

print("-------------------------------------")

# check distribution of the target (label) value
print("Target Distribution")
print("1 - Heart Disease Positive")
print("0 - Heart Disease Negative")
print(data['target'].value_counts())

print("-------------------------------------")

# separate the target(label) and the features
X = data.drop(columns='target', axis=1)
Y = data['target']
print("Features:")
print(X)
print("Target (Label):")
print(Y)

print("-------------------------------------")

# split the data into training:test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)


# train the model using logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)

filename = './heart_disease.pkl'
joblib.dump(model, filename)

# Load the model from the file
model2 = joblib.load("./heart_disease.pkl")

print("-------------------------------------")

# Model Evalutation with accuracy (training data)
X_train_prediction = model2.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data:")
print(training_data_accuracy)

# Model Evaluation with accuracy (test_data)
X_test_prediction = model2.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on testing data:")
print(testing_data_accuracy)

print("-------------------------------------")

# building a predictive system
input_data = (42, 1, 1, 120, 295, 0, 1, 162, 0, 0, 2, 0, 2)
input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1, -1)
prediction = model2.predict(input_data_reshaped)
print("-------------------------------------")
if(prediction[0] == 0):
    print("Person is Heart Disease Negative")
else:
    print("Person is Heart Disease Positive")
