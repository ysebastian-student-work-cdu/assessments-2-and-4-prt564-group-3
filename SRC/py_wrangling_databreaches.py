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

#=========  Linear Regression(week-1) =============
print("======================================== Multiple Linear regression model and output(Week-1)===========================================")
#Separate explanatory variables (x) from the response variable (y)
#x = df.iloc[:,[19,26]].values

# x columns are BreachDateYear,IsVerified(0,1),IsFabricated(0,1),IsSensitive(0,1),IsRetired(0,1),IsSpamList(0,1),IsMalware(0,1),email_addresses	
# ,passwords,usernames,names,ip_addresses,phone_numbers,date_of_birth,physical_addresses,genders,geographic locations[these are top 10 breachdata]
#x = df_wrangled.iloc[:,[19,26,27,28,29,30,31,70,116,158,104,94,122,59,124,82,83]]
print("---OPTION 2: BreachDateYear, BreachAddYear, BreachModifyYear")
# x columns are BreachDateYear,IsVerified(0,1),IsFabricated(0,1),IsSensitive(0,1),IsRetired(0,1),IsSpamList(0,1),IsMalware(0,1)
x2 = df.iloc[:, [17,18,19,20,21,22,23,24,25] + list(range(26, 32))]
#print(x2)
y2 = df.iloc[: , [7]]
#print(y)

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train2, y_train2)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred2 = model.predict(X_test2)
# print(y_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test2, y_pred2)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test2, y_pred2)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test2, y_pred2))
# Normalised Root Mean Square Error
y_max2 = y2.max()
y_min2 = y2.min()
rmse_norm2 = rmse / (y_max2 - y_min2)

# R-Squared
r_2 = metrics.r2_score(y_test2, y_pred2)

print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm2)
print("R^2: ", r_2)

print("---OPTION 1: BreachDateYear-----")
# x columns are BreachDateYear,IsVerified(0,1),IsFabricated(0,1),IsSensitive(0,1),IsRetired(0,1),IsSpamList(0,1),IsMalware(0,1)
x = df.iloc[:, [17,18,19] + list(range(26, 32))]
#print(x)
y = df.iloc[: , [7]]
#print(y)

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
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
# print(y_pred)

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

print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

#LINEAR REGRESSION MODEL WITH DATA: BREACHDATE, ISVERIFIED.....
print("====Linear regression model with Original data==")

x = df.iloc[:, [17,18,19,26,27,28,29,30,31]]
#print(x)
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# Build a linear regression model
X_train = sm.add_constant(X_train)  # this ensure that intercept is calculated
model = sm.OLS(y_train,X_train).fit()


# Print model details
model_details = model.summary()
print(model_details)

#FORWARD SELECTION

print("RUN THE BEST VARIABLE FOR FORWARD SELECTION WITH GRAPH")
x = df.iloc[:, [17,18,19,26,27,28,29,30,31]]
#print(x)
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
model = LinearRegression()
model_sfs_best = sfs(model, k_features="best", forward=True, verbose=2, scoring='r2')       #this will check all the columns and take time
model_sfs_best = model_sfs_best.fit(X_train, y_train)

plot_sfs(model_sfs_best.get_metric_dict(),kind='std_err')
plt.title("Sequential Forward Selection (w. StdErr)")
plt.ylabel("Performance (R^2)")
plt.grid()
plt.show()

print("\n Use Forward feature selection to get top 4 explanatory variables")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
"""
PERFORM FORWARD VARIABLE SELECTION
"""
# Build a linear regression model
model = LinearRegression()

#perform forward selection
model_sfs = sfs(model, k_features=4, forward=True, verbose=2, scoring='r2')    #this will take any 10 columns and take time to compile(ideal value is 5)
#model_sfs = sfs(model, k_features="best", forward=True, verbose=2, scoring='r2') 
model_sfs = model_sfs.fit(X_train, y_train)
# Get top-5 selected explanatory variables
feat_names = list(model_sfs.k_feature_names_)
print('\n TOP 4 VARIABLES \n',feat_names)


"""
REBUILD THE MODEL USING THE SELECTED VARIABLES
"""

# Include the name of the response variable
feat_names.append("PwnCount")

# Reduce the original dataset to top-5 selected variables, plus the response variable
df_fwd = df[feat_names]

# (Again) Separate explanatory variables (x) from the response variable (y)
x = df_fwd.iloc[:,:-1].values
#print(x)
y = df_fwd.iloc[:,-1].values
#print(y)

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Rebuild a linear regression model 
model = LinearRegression()

# Fit the model using the selected variables
model.fit(X_train, y_train)
#print(model)
# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_train)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_train, "Predicted": y_pred})
print(df_pred)


# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_train, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_train, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_train, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_train, y_pred)


print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

print("====Linear regression model with 4 SELECTION data==")

x = df.iloc[:, [19,26,28,31]]
#print(x)
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# Build a linear regression model
X_train = sm.add_constant(X_train)  # this ensure that intercept is calculated
model = sm.OLS(y_train,X_train).fit()


# Print model details
model_details = model.summary()
print(model_details)

print("===Metrics with 4 variables===")
# Use linear regression to predict the values of (y) in the test set

x = df.iloc[:, [19,26,28,31]]
#print(x)
y = df.iloc[: , [7]]
# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
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

print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

# ORIGINAL: compare the cumulative explained variance versus number of PCA components
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
print("Dimension of original data:", X_train.shape)
print("Dimension of PCA-reduced data:", X_train_p.shape)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train_p, y_train)

# Use linear regression to predict the values of (y) in the training set
y_pred = model.predict(X_train_p)

# Get R-Squared score
r_2 = metrics.r2_score(y_train, y_pred)
print("Trained with 4-component PCA:")
print("R^2: ", r_2)

print("======Perform standardisation with 4 variables====")
#perform standardisation
X_train_std = StandardScaler().fit_transform(X_train)
#build PCA
pca = PCA(n_components=4)
X_train_p = pca.fit_transform(X_train_std)
print ("==============================PCA_X-train shape =============================")
print(X_train.shape)
print(X_train_p.shape)

model = LinearRegression()
model.fit(X_train_p,y_train)

y_pred = model.predict(X_train_p)
r_2 = metrics.r2_score(y_train,y_pred)
print("Trained with 4-component PCA after standardisation:")
print("R^2: ", r_2)

print("-------------------------COMPARE MODEL----------------------------------")
##################################COMPARE AMONG MODEL #############################################
print("====BUILD AND EVALUATE LINEAR REGRESSION MODEL WITH 4 VARIABLE======")

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, [19,26,28,31]]
y = df.iloc[: , [7]]

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
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

print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

print("====BUILD AND EVALUATE LINEAR REGRESSION MODEL WITH 4 VARIABLE apply PCA only======")

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, [17,18,19,26,27,28,29,30,31]]
#print(x)
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# Build the PCA on the training set (X_Train - orginal varaiable 9)
pca = PCA().fit(X_train)

# Train a linear regression on PCA-transformed training data (top-4 components)
pca = PCA(n_components=4)
X_train_p = pca.fit_transform(X_train)

# Compare the dimensionality of the original data vs. its dimensionality reduced version
print("Dimension of original data:", X_train.shape)
print("Dimension of PCA-reduced data:", X_train_p.shape)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train_p, y_train)

# Use linear regression to predict the values of (y) in the training set
y_pred = model.predict(X_train_p)

# Get R-Squared score
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
print("Trained with 4-component PCA:")
print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

print("====BUILD AND EVALUATE LINEAR REGRESSION MODEL WITH 4 VARIABLE apply standardisation and PCA ======")
# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, [17,18,19,26,27,28,29,30,31]]
#print(x)
y = df.iloc[: , [7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

X_train_std = StandardScaler().fit_transform(X_train)
#build PCA
pca = PCA(n_components=4)
X_train_p = pca.fit_transform(X_train_std)

model = LinearRegression()
model.fit(X_train_p,y_train)

y_pred = model.predict(X_train_p)
# Get R-Squared score
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
print("Trained with 4-component PCA after standardisation:")
print("MLR performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

## Standardisation

# separate response variable (y) from explanatory variables (X)
x = df.iloc[:, [19,26,28,31]]
y = df.iloc[: , [7]]

# training and test splits on original data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# apply standardisation to explanatory variables
std_scaler = preprocessing.StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)


# build two types of Naive Bayes classifiers for comparison

print("### without PCA ###")

# on original data
clf = GaussianNB().fit(X_train, y_train)
y_hat = clf.predict(X_train)
print("\nPrediction accuracy for the training dataset:")
print("{:.2%}".format(metrics.accuracy_score(y_train, y_hat)))

# on standardised data
clf_std = GaussianNB().fit(X_train_std, y_train)
y_hat_std = clf_std.predict(X_train_std)
print("\nPrediction accuracy for the training dataset (with standardisation):")
print("{:.2%}".format(metrics.accuracy_score(y_train, y_hat_std)))

# build two other of Naive Bayes classifiers 
# (with prior dimensionality reduction by PCA)
print("\n\n### with PCA ###")

# initialise 2-component PCA
pca = PCA(n_components=2)

# PCA on original explanatory variables
X_train = pca.fit_transform(X_train)

# PCA on standardised explanatory variables
X_train_std = pca.fit_transform(X_train_std)

# on original data
clf = GaussianNB().fit(X_train, y_train)
y_hat = clf.predict(X_train)
print("\nPrediction accuracy for the training dataset:")
print("{:.2%}".format(metrics.accuracy_score(y_train, y_hat)))

# on standardised data
clf_std = GaussianNB().fit(X_train_std, y_train)
y_hat_std = clf_std.predict(X_train_std)
print("\nPrediction accuracy for the training dataset (with standardisation):")
print("{:.2%}".format(metrics.accuracy_score(y_train, y_hat_std)))

