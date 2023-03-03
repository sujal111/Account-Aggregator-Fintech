import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import seed,sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, roc_curve, auc,precision_score
from sklearn.ensemble import RandomForestClassifier

st.title("Classifying Fraudulent and Valid Transactions")


data = pd.read_csv('static/fraud.csv')

st.title("Dataset Overview")
st.write(data)

st.write("""step - integer - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - string/categorical - type of transaction: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - float - amount of the transaction in local currency.

nameOrig - string - customer who initiated the transaction

oldbalanceOrg - float initial balance before the transaction

newbalanceOrig - float - new balance after the transaction

nameDest - string - customer who is the recipient of the transaction

oldbalanceDest - float - initial balance of recipient before the transaction.

newbalanceDest - float - new balance of recipient after the transaction.

isFraud - boolean/binary - determines if transaction is fraudulent (encoded as 1) or valid (encoded as 0)

isFlaggedFraud - boolean/binary - determines if transaction is flagged as fraudulent (encoded as 1) or not flagged at all (encoded as 0). An observation is flagged if the transaction is fraudulent and it involved a transfer of over 200,000 in the local currency.""")





data.pop('Unnamed: 0')

correlation=data.corr()

st.write("Correlation Value for isFraud")
st.write(correlation['isFraud'])

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

from sklearn.model_selection import train_test_split

x=np.array(data[['type','amount','oldbalanceOrg','newbalanceOrig']])
y=np.array(data[['isFraud']])

from sklearn.tree import DecisionTreeClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
st.title("Decison Tree accuracy on test data :-")
st.write(model.score(x_test,y_test))

st.write("Lets try to take info from bob api")
st.write("This is the detail of one of the transaction")
st.write("Type Amount Balance")

st.write("DEBIT	10070.00	51070.50")	

features = np.array([[5, 10070.00, 51070.50, 41000.50]])
st.write("Test data")
st.write(features)
st.write(model.predict(features))
