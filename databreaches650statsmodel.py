import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split


# Read dataset into a DataFrame
df = pd.read_csv("boston.csv")


# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split dataset into 60% training and 40% test sets 
# This step ensures that the statsmodels' model is trained on the same training set as scikit-learn's
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# Build a linear regression model
X_train = sm.add_constant(X_train)  # this ensure that intercept is calculated
model = sm.OLS(y_train,X_train).fit()


# Print model details
model_details = model.summary()
print(model_details)


