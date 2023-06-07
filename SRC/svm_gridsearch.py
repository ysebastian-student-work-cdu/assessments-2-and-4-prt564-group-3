from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 


# Load the data

df = pd.read_csv("databreaches650_final.csv")
# Convert dataframe to dataset
X = df.iloc[:, :-1]  # features
y = df.iloc[:, -1]   # target

X = X.values
y = y.values

# # splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 
                     'C': [1, 10, 100, 1000]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring=score
    )
    
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_, "\n")
    
    print("Grid scores on development set:\n")
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std, params))
    print()

    print("Detailed classification report:\n")

    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
  
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    
# Build an SVM model
# create an instance of the SVM classifier with a Linear kernel
model = svm.SVC(kernel='linear', probability=True)

# train the model using the training set
model.fit(X_train, y_train)

# predict the classes in the test set
y_pred = model.predict(X_test)

# Evaluate the model
# accuracy
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))

# precision
print("Precision: %.3f " % metrics.precision_score(y_test, y_pred, pos_label=0))

# recall
print("Recall: %.3f" % metrics.recall_score(y_test, y_pred, pos_label=0))

# F1 (F-Measure)
print("F1: %.3f" % metrics.f1_score(y_test, y_pred, pos_label=0))


# compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# pretty print Confusion Matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y)
disp.plot()
plt.show()