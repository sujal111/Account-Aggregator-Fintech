import streamlit as st

st.markdown("# Main page")
st.sidebar.markdown(" :)( ")

st.title("Bank Of Baroda Hackathon")
  
st.header("1. Customer Segmentation")
st.write("Dataset contains the following information")
st.write("Gender :- which can be predicted using name or can be directly access")
st.write("Age :- Age can be calculated as we have the date of birth of each customer")
st.write("Annual income :- it helps us to divide the customer according to thier earning capacity which is directly proportional to the amount they can spend , we already have estimated salary/month which can be used to get this information")
st.write("Spending score:- This is a direct way to calculate spending power/capacity of a person A spending score is a metric used to determine how likely a customer is to make a purchase based on their spending habits.")
st.write("Spending Score = (Number of Transactions) x (Average Transaction Value)")

st.write("This formula takes into account both the frequency of the customer's purchases and the amount of money they typically spend in each transaction. You can then use the resulting spending score to identify which customers are most valuable to your business and tailor your marketing and sales efforts accordingly.")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

@st.cache(allow_output_mutation=True)
def data_load():
    customer_data = pd.read_csv('static/Customers-segmentation-dataset.csv')
    return customer_data

data = data_load()

st.write("Dataset overview - This dataset contains factors based on which spending score is calculated") 
st.write(data)
st.write("stats overview")
st.write(data.describe())

fig = plt.figure(figsize=(40,30))
sns.distplot(data['Age'],color= 'green',bins=20)
plt.title('Age distribution plot', fontsize = 20)
plt.xlabel('Age', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
st.pyplot(fig)

fig = plt.figure(figsize=(20,10))
sns.boxplot(data['Age'])
plt.title('Age box plot', fontsize = 15)
plt.xlabel('Age', fontsize = 12)
st.pyplot(fig)

fig = plt.figure(figsize=(20,10))
sns.distplot(data['Annual Income (k$)'],color= 'blue',bins=20)
plt.title('Annual Income distribution plot', fontsize = 15)
plt.xlabel('Annual Income', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
st.pyplot(fig)

fig = plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot = True)
st.pyplot(fig)

fig = plt.figure(figsize=(20,10))

Income_Spend = data[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(Income_Spend)
    wcss.append(km.inertia_)

fig = plt.figure(figsize=(15,8))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Graph', fontsize = 15)
plt.xlabel('No. of Clusters', fontsize = 12)
plt.ylabel('wcss', fontsize = 12)
st.pyplot(fig) 


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
# return a label for each data point based on their cluster
y_means = kmeans.fit_predict(Income_Spend)
print(y_means)

fig = plt.figure(figsize=(15,8))
plt.scatter(Income_Spend[y_means == 0, 0], Income_Spend[y_means == 0, 1], s = 100, c = 'pink', label = 'Average')
plt.scatter(Income_Spend[y_means == 1, 0], Income_Spend[y_means == 1, 1], s = 100, c = 'yellow', label = 'Spenders')
plt.scatter(Income_Spend[y_means == 2, 0], Income_Spend[y_means == 2, 1], s = 100, c = 'cyan', label = 'Best')
plt.scatter(Income_Spend[y_means == 3, 0], Income_Spend[y_means == 3, 1], s = 100, c = 'magenta', label = 'Low Budget')
plt.scatter(Income_Spend[y_means == 4, 0], Income_Spend[y_means == 4, 1], s = 100, c = 'orange', label = 'Saver')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.legend()
plt.title('Customere Segmentation using Annual Income and Spending Score', fontsize = 15)
plt.xlabel('Annual Income', fontsize = 12)
plt.ylabel('Spending Score', fontsize = 12)
st.pyplot(fig)

Age_Spend = data[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(Age_Spend)
    wcss.append(km.inertia_)

fig = plt.figure(figsize=(15,8))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 15)
plt.xlabel('No. of Clusters', fontsize = 12)
plt.ylabel('wcss', fontsize = 12)
st.pyplot(fig) 

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ymeans = kmeans.fit_predict(Age_Spend)

fig = plt.figure(figsize=(15,8))
plt.scatter(Age_Spend[ymeans == 0, 0], Age_Spend[ymeans == 0, 1], s = 100, c = 'pink', label = 'Regular Customers' )
plt.scatter(Age_Spend[ymeans == 1, 0], Age_Spend[ymeans == 1, 1], s = 100, c = 'orange', label = 'Young Targets')
plt.scatter(Age_Spend[ymeans == 2, 0], Age_Spend[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Usual Customer')
plt.scatter(Age_Spend[ymeans == 3, 0], Age_Spend[ymeans == 3, 1], s = 100, c = 'red', label = 'Old Targets')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 50, c = 'black')
plt.legend()
plt.title('Customere Segmentation using Annual Income and Spending Score', fontsize = 15)
plt.xlabel('Age', fontsize = 12)
plt.ylabel('Spending Score', fontsize = 12)
st.pyplot(fig)


data["Total"] = data["Annual Income (k$)"] + data["Spending Score (1-100)"]
Income_Spend = data[['Age' ,'Total']].iloc[: , :].values
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(Income_Spend)
    wcss.append(km.inertia_)

fig = plt.figure(figsize=(15,8))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Graph', fontsize = 15)
plt.xlabel('No. of Clusters', fontsize = 12)
plt.ylabel('wcss', fontsize = 12)
st.pyplot(fig) 


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
# return a label for each data point based on their cluster
y_means = kmeans.fit_predict(Income_Spend)
print(y_means)

fig = plt.figure(figsize=(15,8))
plt.scatter(Income_Spend[y_means == 0, 0], Income_Spend[y_means == 0, 1], s = 100, c = 'pink', label = 'Middle Segment')
plt.scatter(Income_Spend[y_means == 1, 0], Income_Spend[y_means == 1, 1], s = 100, c = 'yellow', label = 'Higher Segment')
plt.scatter(Income_Spend[y_means == 2, 0], Income_Spend[y_means == 2, 1], s = 100, c = 'cyan', label = 'Lower Segment')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.legend()
plt.title('Customere Segmentation using Annual Income + Spending Score', fontsize = 15)
plt.xlabel('Age', fontsize = 12)
plt.ylabel('Annual Income+Spending Score', fontsize = 12)
st.pyplot(fig)

