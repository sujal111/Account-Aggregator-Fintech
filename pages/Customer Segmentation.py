import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import mixture
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

st.title("Customer Segmentation")

st.write("Based on amount spending in each categories")

@st.cache(allow_output_mutation=True)
def data_load():
    customer_data = pd.read_csv('static/the_data_cluster.csv')
    customer_data.drop(['Channel','Region'],axis=1,inplace=True)
    return customer_data

data = data_load()

st.write(data)

scaled_data = data.apply(lambda x: np.log(x))
clean_data = scaled_data
principal = PCA()
principal.fit(clean_data)

fig = plt.figure(figsize =(10,8))
plt.plot(range(1,7), principal.explained_variance_ratio_.cumsum(),marker ="o", linestyle = "--")
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
st.pyplot(fig)

principal_n = PCA(n_components = 4)
principal_n.fit(clean_data)
pca_scaled_data = principal_n.transform(clean_data)


df_k_Means = data.copy()

sse = []

kmeans_pca = KMeans(n_clusters = 3, init="k-means++",random_state = 42)
kmeans_pca.fit(pca_scaled_data)

final_df_k_Means = pd.DataFrame(pca_scaled_data)
final_df_k_Means.columns = ["Component 1","Component 2","Component 3","Components 4"]

df_k_Means['Cluster'] = kmeans_pca.labels_

NumofClusters_K_Means = df_k_Means['Cluster'].unique()           

idxCluster1= df_k_Means[df_k_Means["Cluster"] == 0].index
idxCluster2= df_k_Means[df_k_Means["Cluster"] == 1].index
idxCluster3= df_k_Means[df_k_Means["Cluster"] == 2].index

df1 = df_k_Means.loc[idxCluster1]
fig = plt.figure(figsize =(10,8))
st.write("Variance of Components")

plt.title('Cluster-0 - K_Means')
plt.xticks(rotation=90)
plt.boxplot(df1,labels=list(df1.columns))
st.pyplot(fig)
st.write("The first cluster created seems to contain customers that on average spend more on Fresh, Grocery and Milk products while their spending on other categories is relatively low.")

#==============================BoxPlot_C2==============================#
df2 = df_k_Means.loc[idxCluster2]
fig = plt.figure(figsize =(10,8))


plt.title('Cluster-1 - K_Means')
plt.xticks(rotation=90)
plt.boxplot(df2,labels=list(df2.columns))
st.pyplot(fig)
st.write("The second cluster created contains costumers that spend a lot of money on Fresh products but overall, their spendings are low. The costumers of this cluster, on average, spend small amounts of money on goods, except from Fresh products.")

#===============================BoxPlot_C3=============================#
df3 = df_k_Means.loc[idxCluster3]
fig = plt.figure(figsize =(10,8))
plt.title('Cluster-2 - K_Means')
plt.xticks(rotation=90)
plt.boxplot(df3,labels=list(df3.columns))
st.pyplot(fig)
st.write("The third cluster contains costumers that spend a lot of money on Grocery, Milk, and Detergent Paper while their spendings on the other products are relatively low. The customers of this cluster, in general, have a more balanced spending routine. Overall, the costumers of the different clusters have similar spending routines on some products (for instance Cluster #0 seems to spend a lot on Fresh while the same happens for the costumers of Cluster #1) but at the same time they are differentiated on their spendings on other products. So, the clusters created could offer valuable insights on the different behaviors.")



df_DBSCAN = data.copy()
final_df_DBSCAN = pd.DataFrame(pca_scaled_data)
final_df_DBSCAN.columns = ["Component 1","Component 2","Component 3","Component 4"]
#========================Optimizing_Parameters===========================#
neighbors = NearestNeighbors(n_neighbors = 2*6) #===2*Dimension====#
neighbors_fit = neighbors.fit(final_df_DBSCAN)
distances, indices = neighbors_fit.kneighbors(final_df_DBSCAN)

distances =np.sort(distances, axis=0)
distances = distances[:,1]
fig = plt.figure(figsize =(10,8))
plt.title("Distances - Points")
plt.xlabel("Points")
plt.ylabel("Distances")
plt.plot(distances)  
st.pyplot(fig)     #Best value for ε seems to be around 1.2#

#======================DBSCAN=#minPts=7_AND_ε==1.2======================#
dbscan = DBSCAN(eps = 1.2,min_samples=7).fit(final_df_DBSCAN)
labels = dbscan.labels_

final_df_DBSCAN["Cluster-Segment"] = labels

df_DBSCAN['Cluster'] = labels

NumofClusters_DBSCAN = df_DBSCAN["Cluster"].unique()



idxCluster1= df_DBSCAN[df_DBSCAN["Cluster"] == -1].index
idxCluster2= df_DBSCAN[df_DBSCAN["Cluster"] == 0].index
idxCluster3= df_DBSCAN[df_DBSCAN["Cluster"] == 1].index
#===============================BoxPlots================================#
#================================BoxPlot_C1=============================#
df7 = df_DBSCAN.loc[idxCluster1]
fig = plt.figure(figsize =(10,8))
plt.title('Cluster- -1 - DBSCAN')
plt.xticks(rotation=90)
plt.boxplot(df7,labels=list(df7.columns))
st.pyplot(fig)
st.write("The first cluster created by the DBSCAN algorithm seems to contain customers that on average spent average to low amounts of money on products with an exception in Grocery products. Spendings on Delicassen, Frozen are minimum.")
#===============================BoxPlot_C2==============================#
df8 = df_DBSCAN.loc[idxCluster2]
fig = plt.figure(figsize =(10,8))
plt.title('Cluster- 0 - DBSCAN')
plt.xticks(rotation=90)
plt.boxplot(df8,labels=list(df8.columns))
st.pyplot(fig)
st.write("The second cluster created, seems to contain costumers with high spendings on Fresh products while spendings on other goods are average to low.")
#===============================BoxPlot_C3==============================#
df14 = df_DBSCAN.loc[idxCluster3]
fig = plt.figure(figsize =(10,8))
plt.title('Cluster- 1 - DBSCAN')
plt.xticks(rotation=90)
plt.boxplot(df14,labels=list(df14.columns))
st.pyplot(fig)
st.write("The third cluster created contains customers with high spendings on Grocery and relatively high spendings on Detergent Paper and Milk products while spendings on the other goods are low to minimum. Overall, from this method it is not possible to group clusters together. Cluster #1 seems to be differentiated because of the relative big spendings on some products. While Cluster #-1 and #0 cannot be differentiated easily from each other, Cluster #-1 contains, on average, bigger spenders in comparison to Cluster #0.")
#===============================Tree====================================#
