import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'/Users/bannusagi/Downloads/logit classification.csv')

X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

varience = classifier.score(X_test, y_test)
print(varience)

bias = classifier.score(X_train, y_train)
print(bias)

#future prediction

df1 = pd.read_csv(r'/Users/bannusagi/Downloads/Future prediction1.csv')
d2 = df1.copy()

df1 = df1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df1 = ss.fit_transform(df1)

y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifier.predict(df1)