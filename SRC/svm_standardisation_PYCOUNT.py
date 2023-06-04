import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler,KBinsDiscretizer, OneHotEncoder
from sklearn import datasets, preprocessing, svm, metrics
from sklearn.model_selection import cross_validate, cross_val_score,KFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Read dataset into a DataFrame

# Read dataset into a DataFrame
df= pd.read_csv('c:/temp/anna/out2.csv')
# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
"""" 
SVM classifier Method"""
print("=================SVM CLASSIFIER METHOD===============")

# SVM with no standardisation
clf = SVC(kernel='linear').fit(X_train, y_train)
y_hat = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_hat)


# create SVM with standardisation via standardisation --> classification pipeline
pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
pipe.fit(X_train, y_train)
acc_std = pipe.score(X_test, y_test)

# compare accuracy scores
print("Accuracy comparison:")
print("SVM --> %.3f%%" % (acc * 100))
print("SVM with standardisation --> %.3f%%" % (acc_std * 100))
