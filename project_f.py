
import pandas as pd
import numpy as np


dt=pd.read_csv("Crop_Recommendation.csv")
x=dt.iloc[:,:-1].values
y=dt.iloc[:,-1].values

dt.describe()

print(x)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

import matplotlib.pyplot as plt
import seaborn as sns

numeric_data = dt.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=dt[feature], color='skyblue')
    plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1)

from sklearn.metrics import confusion_matrix,accuracy_score
cnf=confusion_matrix(y_test,y_pred)
print(cnf)

accuracy_score(y_test,y_pred)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score : ", f1)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='weighted')
print(precision)

n=float(input("Enter the ratio of Nitrogen in soil : "))
p=float(input("Enter the ratio of Phosphorous in soil : "))
k=float(input("Enter the ratio of Potassium in soil : "))
t=float(input("Enter the Temperature in celsius : "))
h=float(input("Enter th Humidity in % value : "))
p=float(input("Enter the pH Value : "))
r=float(input("Enter the rainfall in cm : "))
X=np.array([[n,p,k,t,h,p,r]])
X=sc.transform(X)
Y=rf.predict(X)
print("Most Suitable Crop for your Conditions is : "+lb.inverse_transform(Y))