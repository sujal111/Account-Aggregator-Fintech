import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Good Loans and Bad loans")
df = pd.read_csv("static/accepted_2007_to_2018Q4.csv")

st.title("Dataset Overview For Good/Bad Loan")
st.write(df)


st.title("Dataset's Features Overview For Good/Bad Loan")
st.write("'term' : The number of payments on the loan, where values are in months and can be either 36 or 60.")
st.write("'int_rate :  The interest rate on the loan")
st.write("'sub_grade  : Assigned loan subgrade score based on borrower's credit history")
st.write("'emp_length': Borrow's employment length in years.")
st.write("'dti' : A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage, divided by the borrower's monthly income")
st.write("'mths_since_recent_inq': Months since most recent inquiry")
st.write("'revol_util' : Revolving line utilization rate, or the amount of credit the borrower uses relative to all available revolving credit")
st.write("'bc_util': Ratio of total current balance to high credit/credit limit for all bankcard accounts")
st.write("'num_op_rev_tl' : Number of open revolving accounts")

st.title("observation")
st.write("We have a lot of features but we got this top 9 features using Logistic Regression with SequentialFeatureSelector")






df['loan_status'].value_counts()

df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]


final_features = ['term','int_rate','sub_grade','emp_length','dti','mths_since_recent_inq','revol_util','bc_util','num_op_rev_tl','loan_status']

df = df[final_features]

df_temp = df.copy()



df_temp['term'] = df_temp['term'].apply(lambda x: int(x[0:3]))
df_temp['loan_status'] = df_temp['loan_status'].map({'Fully Paid':1,'Charged Off':0})

mapp = {'< 1 year':0.5,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}
df_temp['emp_length'] = df_temp['emp_length'].map(mapp)

df_temp['emp_length'] = df_temp['emp_length'].fillna(value=0)
df_temp['term'] = df_temp['term'].fillna(value=0)
df_temp['int_rate'] = df_temp['int_rate'].fillna(value=0)
df_temp['dti'] = df_temp['dti'].fillna(value=0)
df_temp['mths_since_recent_inq'] = df_temp['mths_since_recent_inq'].fillna(value=0)
df_temp['revol_util'] = df_temp['revol_util'].fillna(value=0)
df_temp['bc_util'] = df_temp['bc_util'].fillna(value=0)
df_temp['num_op_rev_tl'] = df_temp['num_op_rev_tl'].fillna(value=0)

df_temp = df_temp[df_temp['emp_length'] > 0]
df_temp = pd.get_dummies(df_temp)

st.title("Preprocessed Data + Feature Engineering")
st.write(df_temp)

from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix

X = df_temp.drop(["loan_status"],axis=1)
y = df_temp["loan_status"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


fig = plt.figure(figsize=(20,25))
sns.heatmap(df.corr())

st.title("HeatMap for the different features")
st.pyplot(fig)

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.title("We have built different ML models to see their performance")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
pred_lr = model.predict(X_test)
st.write("Logistic Regression")
st.write('Accuracy of Logistic Regression: {:.2f}'.format(accuracy_score(pred_lr,y_test)*100))
st.write("*"*20)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)
pred_lr = model.predict(X_test)
st.write("RandomForestClassifier")
st.write('Accuracy of RandomForestClassifier: {:.2f}'.format(accuracy_score(pred_lr,y_test)*100))
st.write("*"*20)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train,y_train)
pred_lr = model.predict(X_test)
st.write("KNeighborsClassifier")
st.write('Accuracy of KNeighborsClassifier: {:.2f}'.format(accuracy_score(pred_lr,y_test)*100))
st.write("*"*20)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=200)
model.fit(X_train,y_train)
pred_lr = model.predict(X_test)
st.write("GradientBoostingClassifier")
st.write('Accuracy of GradientBoostingClassifier: {:.2f}'.format(accuracy_score(pred_lr,y_test)*100))
