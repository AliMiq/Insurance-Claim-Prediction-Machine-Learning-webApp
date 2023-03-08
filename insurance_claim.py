
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns

data=pd.read_csv("insurance3r2.csv")

data.head()

data.info()

data.describe()

data=data.dropna()

plt.title('Class Distributions \n (0: No Claim || 1: Claim)', fontsize=14)
sns.set(style="darkgrid")
sns.countplot(data['insuranceclaim'])
plt.grid()
plt.show()

corr=data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True)
plt.show()

data=data.drop('region',axis=1)

data.head()

plt.figure(figsize = (16, 8))
sns.barplot(x = 'age', y = 'charges', data = data)

plt.title("Age vs Charges")

plt.figure(figsize = (6, 6))
sns.barplot(x = 'sex', y = 'charges', data = data)

plt.title('sex vs charges')

plt.figure(figsize = (12, 8))
sns.barplot(x = 'children', y = 'charges', data = data)

plt.title('children vs charges')

plt.figure(figsize = (6, 6))
sns.barplot(x = 'smoker', y = 'charges', data = data)
plt.title('smoker vs charges')

X=data.iloc[:,:-1]
X.head()

X.shape

Y=data.iloc[:,-1]
Y.head()

Y.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
data.to_csv('finaldata.csv')
X_test.to_csv('testing.csv')

from sklearn.preprocessing import StandardScaler

#ss=StandardScaler()

#X_train=ss.fit_transform(X_train)

#X_train

#X_train=pd.DataFrame(X_train,columns=X_test.columns)

#X_train

#X_test=ss.fit_transform(X_test)

#X_test=pd.DataFrame(X_test,columns=X_train.columns)

#X_test

#y_train


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train, y_train)
ypred=rf.predict(X_test)
print(confusion_matrix(y_test,ypred))


from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=rf,X=X_train,y=y_train,cv=10)
acc.mean()
acc.std()

import pickle
# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


