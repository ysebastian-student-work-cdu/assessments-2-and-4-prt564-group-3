import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import requests                    
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# read the data

df =  pd.read_csv('databreaches650_prepared.csv')
print("OPTION 1: Linear regression model with 4 features")

# Choose 4 features of 'year','isverified','issensitive','ismalware' to predict the pwncount
x = df.iloc[:, [19,26,28,31]]
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
X_train = sm.add_constant(X_train)  # this ensure that intercept is calculated
model = sm.OLS(y_train,X_train).fit()

# Print model details
model_details = model.summary()
print(model_details)

print("Metrics with 4 variables")

# Use linear regression to predict the pwncount (values of (y)) in the test set 
x = df.iloc[:, [19,26,28,31]]
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
#df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)
print("OPTION 1: Trained with 4 original variables :")
print("R^2: ", r_2)


# Apply PCA

# ORIGINAL: Choose 9 features to predict the pwncount 
# compare the cumulative explained variance versus number of PCA components
x = df.iloc[:, [17,18,19,26,27,28,29,30,31]]
#print(x)

y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build the PCA on the training set (X_Train - orginal varaiable 9)
pca = PCA().fit(X_train)

# Plot the cumulative explained variance versus number of PCA components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xticks(range(1,10))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

# Train a linear regression on PCA-transformed training data (top-4 components)
pca = PCA(n_components=4)
X_train_p = pca.fit_transform(X_train)

# Compare the dimensionality of the original data vs. its dimensionality reduced version
#print("Dimension of original data:", X_train.shape)
#print("Dimension of PCA-reduced data:", X_train_p.shape)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train_p, y_train)

# Use linear regression to predict the values of (y) in the training set
y_pred = model.predict(X_train_p)

# Get R-Squared score
r_2 = metrics.r2_score(y_train, y_pred)
print("OPTION 2: Trained with 4-component PCA:")
print("R^2: ", r_2)

#print("Perform standardisation with 4 variables")
#perform standardisation
X_train_std = StandardScaler().fit_transform(X_train)
#build PCA
pca = PCA(n_components=4)
X_train_p = pca.fit_transform(X_train_std)
#print (" PCA_X-train shape")
#print(X_train.shape)
#print(X_train_p.shape)

model = LinearRegression()
model.fit(X_train_p,y_train)

y_pred = model.predict(X_train_p)
r_2 = metrics.r2_score(y_train,y_pred)
print("OPTION 3: Trained with 4-component PCA after standardisation:")
print("R^2: ", r_2)