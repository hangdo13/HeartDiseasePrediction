import pandas as pd
import sklearn

from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

heart = pd.read_csv('heart.csv')
heart = heart.dropna()


y = heart['target'].to_numpy()
X = heart.drop(columns = ['target']).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=8)

normalizer = sklearn.preprocessing.MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


#Logistic Regression Classifier
lg_clf = LogisticRegression(random_state=8)
lg_clf.fit(X_train_norm,y_train)
lg_y_pred = lg_clf.predict(X_test_norm)
print('Logistic Regression')
print('predicted:', lg_y_pred)
print('true:', y_test)
accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = lg_y_pred)
precision = sklearn.metrics.precision_score(y_true=y_test, y_pred = lg_y_pred)
recall = sklearn.metrics.recall_score(y_true=y_test, y_pred = lg_y_pred)
f1 = sklearn.metrics.f1_score(y_true=y_test, y_pred = lg_y_pred)
print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:',recall, 'F1-Score:',f1)


#Decision Tree Classifier
dt_clf = sklearn.tree.DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)
print('Decision Tree')
print('predicted:', dt_y_pred)
print('true:', y_test)
accuracy1 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = dt_y_pred)
precision1 = sklearn.metrics.precision_score(y_true=y_test, y_pred = dt_y_pred)
recall1 = sklearn.metrics.recall_score(y_true=y_test, y_pred = dt_y_pred)
f1_1 = sklearn.metrics.f1_score(y_true=y_test, y_pred = dt_y_pred)
print('Accuracy:', accuracy1, 'Precision:', precision1, 'Recall:',recall1, 'F1-Score:',f1_1)


#RandomForestClassifier
rf_clf = sklearn.ensemble.RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
print('Random Forest')
print('predicted:', rf_y_pred)
print('true:', y_test)
accuracy2 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = rf_y_pred)
precision2 = sklearn.metrics.precision_score(y_true=y_test, y_pred = rf_y_pred)
recall2 = sklearn.metrics.recall_score(y_true=y_test, y_pred = rf_y_pred)
f1_2 = sklearn.metrics.f1_score(y_true=y_test, y_pred = rf_y_pred)
print('Accuracy:', accuracy2, 'Precision:', precision2, 'Recall:',recall2, 'F1-Score:',f1_2)


#Majority Baseline Classifier
y_pred_maj = [mode(y_train)]*len(y_test)
accuracy3 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_maj)
precision3 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_maj)
recall3 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_maj)
f1_3 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_maj)
print('Majority Baseline')
print('Accuracy:', accuracy3, 'Precision:', precision3, 'Recall:',recall3, 'F1-Score:',f1_3)