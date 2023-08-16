import pandas as pd

# load dataset
path = 'data/all_data.csv'
names = ['Answer.quantity', 'Input.quant1']
data = pd.read_csv(path)

#select the data for x and y
x = data.loc[:,names[0]]
y = data.loc[:,names[1]]

#encode the data as dummies
X = pd.get_dummies(x, prefix_sep='_')
Y = pd.get_dummies(y, prefix_sep='_')

#create training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

classifier = OneVsRestClassifier(LogisticRegression(random_state = 0))
classifier.fit(x_train, y_train)

#make predictions and confirm effectivity
y_hat = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

cml = confusion_matrix(y_test, y_hat)
print('Logistic Report:')
print(classification_report(y_test, y_hat))
