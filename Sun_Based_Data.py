## **Upploadind the data**

import pandas as pd
import numpy as np

CustomerData=pd.read_csv('customer_churn_large_dataset.csv')

print("shape of CustomerData : \n",CustomerData.shape)

print(CustomerData.describe())

print(CustomerData.dtypes)

print(CustomerData['Churn'].unique())
### **Data Preprocessing**
# **Filling null values **
No_CustomerID=CustomerData['CustomerID'].isnull()
Order_customerID=range(1,No_CustomerID.sum()+1)
CustomerData.loc[No_CustomerID,'CustomerID']=Order_customerID
CustomerData['CustomerID']=CustomerData['CustomerID'].astype('int64')
CustomerData['Age'].fillna(CustomerData['Age'].mean,inplace=True)
CustomerData['Subscription_Length_Months'].fillna(CustomerData['Subscription_Length_Months'].mean(),inplace=True)
CustomerData['Monthly_Bill'].fillna(CustomerData['Monthly_Bill'].mean(),inplace=True)
CustomerData['Total_Usage_GB'].fillna(CustomerData['Total_Usage_GB'].mean(),inplace=True)
CustomerData['Churn'].fillna(CustomerData['Churn'].mean(),inplace=True)
CustomerData['Churn']=CustomerData['Churn'].round()
print(CustomerData['Churn'].unique())
CustomerData['Name'].fillna('Unknown',inplace=True)
CustomerData['Location'].fillna('Unknown',inplace=True)
CustomerData['Gender'].fillna('Unknown',inplace=True)
## **Model Building**
##**Importing required libraries**
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,classification_report
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
label_encoder=LabelEncoder()
CustomerData['Gender'] = label_encoder.fit_transform(CustomerData['Gender'])
CustomerData['Location'] = label_encoder.fit_transform(CustomerData['Location'])
#**Feature Selection**
X=CustomerData[['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']]
y=CustomerData['Churn']
##Splitting the dataset for training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)
model1=RandomForestClassifier()
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)

#**Model-1:RandomForestClassifier score**
f1=f1_score(y_test,y_pred)
print("F1_Score_M1=",f1)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy_M1=",accuracy)
report=classification_report(y_test,y_pred)
print("Report_M1=",report)

##**Model-2: XGBClassifier score**
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model2=xgb.XGBClassifier()
model2.fit(X_train_scaled,y_train)
y_pred=model2.predict(X_test_scaled)
f1=f1_score(y_test,y_pred)
print("F1_Score_M2=",f1)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy_M2=",accuracy)
report=classification_report(y_test,y_pred)
print("Report_M2=",report)
##**Model-3:DecisionTreeClassifier**
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model3=DecisionTreeClassifier()
model3.fit(X_train,y_train)
y_pred=model3.predict(X_test)
f1=f1_score(y_test,y_pred)
print("F1_Score_M3=",f1)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy_M3=",accuracy)
report=classification_report(y_test,y_pred)
print("Report_M3=",report)


#loading into pickle file

import pickle
filename = 'model.sav'
pickle.dump(model2, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))