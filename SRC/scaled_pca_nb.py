import pandas as pd 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.decomposition import PCA


# read a remote .csv file
df = pd.read_csv('databreaches650_selected.csv', header=0)

# separate response variable (y) from explanatory variables (X)
X = df.iloc[:,1:].values
y = df.iloc[:,0].values

# training and test splits on original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

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