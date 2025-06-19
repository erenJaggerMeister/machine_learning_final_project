import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid

df = pd.read_csv('../Students Social Media Addiction.csv')

# Hal yang dikerjakan adalah untuk mengetahui adiksi pada setiap umur

# untuk pengerjaan k-means
cdf_1 = df[['Age','Addicted_Score']]
cdf_1 = cdf_1.groupby('Age', as_index=False)['Addicted_Score'].mean()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cdf_1)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
cdf_1['Clusters'] = cluster_labels

sil_score = silhouette_score(scaled_data, cluster_labels)
print(f"kmeans --> Silhouette Score: {sil_score:.4f}")

scatter = plt.scatter(cdf_1['Age'],cdf_1['Addicted_Score'], c=cdf_1['Clusters'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Addicted Score')
plt.title('Skor adiksi terhadap penggunaan sosial media berdasarkan umur')
plt.colorbar(scatter,label='Cluster')
plt.grid(True)
plt.show()

# untuk pengerjaan hierarchical clustering 
cdf_2 = df[['Age','Addicted_Score']]
cdf_2 = cdf_2.groupby('Age', as_index=False)['Addicted_Score'].mean()

scaler = StandardScaler()
scaled_data_2 = scaler.fit_transform(cdf_2)

agg = AgglomerativeClustering(n_clusters=3)
cluster_labels_2 = agg.fit_predict(scaled_data_2)
cdf_2['Clusters'] = cluster_labels_2

sil_score_2 = silhouette_score(scaled_data_2, cluster_labels_2)
print(f"Hierarchical Clustering --> Silhouette Score: {sil_score_2:.4f}")

scatter = plt.scatter(cdf_2['Age'],cdf_2['Addicted_Score'], c=cdf_2['Clusters'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Addicted Score')
plt.title('Skor adiksi terhadap penggunaan sosial media berdasarkan umur')
plt.colorbar(scatter,label='Cluster')
plt.grid(True)
plt.show()

# hasil visual silhouette score 
models_silhouette = ['KMeans','Agglomerative']
scores_silhouette = [sil_score,sil_score_2]
colors = ['red','blue']

bars = plt.bar(models_silhouette,scores_silhouette,color=colors)

for bar,score in zip(bars,scores_silhouette):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{score:.4f}', ha='center', va='bottom')

plt.ylim(0, 1)
plt.ylabel('Silhouette Score')
plt.title('Comparison of Clustering Methods using Silhouette Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# hasil yang didapat adalah nilai visualisasi output dari penggunaan k-means dan hierarchical clustering (aglomerartive clustering) 
# nilai yang sama. 
# Data menunjukkan bahwa adiksi terhadap sosial media paling besar ditunjukkan pada usia 18 tahun, sedangkan kecanduan adiksi 
# sosial media dengan nilai terendah ditunjukkan pada usia 22 hingga 24


#cdf_3 akan digunakan untuk ujicoba ketika ada 3 fitur
cdf_3 = df[['Most_Used_Platform','Avg_Daily_Usage_Hours','Addicted_Score']]
print(cdf_3.head())
cdf_3 = cdf_3.groupby('Most_Used_Platform', as_index=False)[['Avg_Daily_Usage_Hours','Addicted_Score']].mean()
print(cdf_3)

# printkan terlebih dahulu data data yang didapatkan 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(len(cdf_3))
y = cdf_3['Avg_Daily_Usage_Hours']
z = cdf_3['Addicted_Score']
labels = cdf_3['Most_Used_Platform']

scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_xlabel('Platform')
ax.set_ylabel('Avg Usage Hours')
ax.set_zlabel('Addicted Score')
ax.set_title('3D Scatter Plot per Platform')

plt.tight_layout()
plt.show()

# lakukan clusterisasi untuk cdf_3 menggunakan k-means 
cdf_4 = cdf_3.copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cdf_4[['Avg_Daily_Usage_Hours', 'Addicted_Score']])

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

cdf_4['Cluster'] = cluster_labels

sil_score_4 = silhouette_score(scaled_features, cluster_labels)
print(f"KMeans (5 clusters) on cdf_4 --> Silhouette Score: {sil_score_4:.4f}")

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(len(cdf_4))  # Posisi kategori pada sumbu x
y = cdf_4['Avg_Daily_Usage_Hours']
z = cdf_4['Addicted_Score']
clusters = cdf_4['Cluster']
labels = cdf_4['Most_Used_Platform']

scatter = ax.scatter(x, y, z, c=clusters, cmap='tab10', s=100)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_xlabel('Platform')
ax.set_ylabel('Avg Usage Hours')
ax.set_zlabel('Addicted Score')
ax.set_title('3D Clustering Result (KMeans, k=5) per Platform')
ax.legend(*scatter.legend_elements(), title="Clusters")

plt.tight_layout()
plt.show()

# lakukan clusterisasi untuk cdf_3 menggunakan agglomerative clustering 
cdf_5 = cdf_3.copy()
scaler = StandardScaler()
scaled_features_5 = scaler.fit_transform(cdf_5[['Avg_Daily_Usage_Hours', 'Addicted_Score']])

agglo = AgglomerativeClustering(n_clusters=5)
cluster_labels_5 = agglo.fit_predict(scaled_features_5)

cdf_5['Cluster'] = cluster_labels_5

sil_score_5 = silhouette_score(scaled_features_5, cluster_labels_5)
print(f"Agglomerative Clustering (5 clusters) on cdf_5 --> Silhouette Score: {sil_score_5:.4f}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(len(cdf_5)) 
y = cdf_5['Avg_Daily_Usage_Hours']
z = cdf_5['Addicted_Score']
clusters = cdf_5['Cluster']
labels = cdf_5['Most_Used_Platform']

scatter = ax.scatter(x, y, z, c=clusters, cmap='tab10', s=100)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_xlabel('Platform')
ax.set_ylabel('Avg Usage Hours')
ax.set_zlabel('Addicted Score')
ax.set_title('3D Clustering Result (Agglomerative, k=5) per Platform')
ax.legend(*scatter.legend_elements(), title="Clusters")

plt.tight_layout()
plt.show()

# ------ Persiapan Dataset (cdf_3) -------
cdf_3 = df[['Most_Used_Platform','Avg_Daily_Usage_Hours','Addicted_Score']]
cdf_3 = cdf_3.groupby('Most_Used_Platform', as_index=False)[['Avg_Daily_Usage_Hours','Addicted_Score']].mean()

# ------ Train-Test Split -------
X = cdf_3[['Avg_Daily_Usage_Hours', 'Addicted_Score']].values
platform_labels = cdf_3['Most_Used_Platform'].values  # untuk pelabelan saja

# Pisahkan data menjadi training dan testing (70%:30%)
X_train, X_test, labels_train, labels_test = train_test_split(X, platform_labels, test_size=0.3, random_state=42)

# ------ KMeans Training -------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=5, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Evaluasi dengan Silhouette Score
train_sil = silhouette_score(X_train_scaled, train_clusters)
test_sil = silhouette_score(X_test_scaled, test_clusters)

print(f"[KMeans] Silhouette Score - Train: {train_sil:.4f}, Test: {test_sil:.4f}")

# ------ Visualisasi KMeans pada Testing Set -------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(len(X_test))
y = X_test[:, 0]
z = X_test[:, 1]

scatter = ax.scatter(x, y, z, c=test_clusters, cmap='tab10', s=100)

ax.set_xticks(x)
ax.set_xticklabels(labels_test, rotation=45, ha='right')
ax.set_xlabel('Platform')
ax.set_ylabel('Avg Usage Hours')
ax.set_zlabel('Addicted Score')
ax.set_title('KMeans Test Clusters (k=5)')
ax.legend(*scatter.legend_elements(), title="Clusters")
plt.tight_layout()
plt.show()

# ------ Agglomerative Training -------
agg = AgglomerativeClustering(n_clusters=5)
train_clusters_agg = agg.fit_predict(X_train_scaled)

# Untuk testing set, AgglomerativeClustering tidak bisa langsung .predict
# jadi gunakan pendekatan retrain lalu assign cluster menggunakan nearest centroid
centroid_model = NearestCentroid()
centroid_model.fit(X_train_scaled, train_clusters_agg)
test_clusters_agg = centroid_model.predict(X_test_scaled)

train_sil_agg = silhouette_score(X_train_scaled, train_clusters_agg)
test_sil_agg = silhouette_score(X_test_scaled, test_clusters_agg)

print(f"[Agglomerative] Silhouette Score - Train: {train_sil_agg:.4f}, Test: {test_sil_agg:.4f}")

# ------ Visualisasi Agglomerative pada Testing Set -------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(len(X_test))
y = X_test[:, 0]
z = X_test[:, 1]

scatter = ax.scatter(x, y, z, c=test_clusters_agg, cmap='tab10', s=100)

ax.set_xticks(x)
ax.set_xticklabels(labels_test, rotation=45, ha='right')
ax.set_xlabel('Platform')
ax.set_ylabel('Avg Usage Hours')
ax.set_zlabel('Addicted Score')
ax.set_title('Agglomerative Test Clusters (k=5)')
ax.legend(*scatter.legend_elements(), title="Clusters")
plt.tight_layout()
plt.show()

# visualisasi hasil training dan testing 

models_result_silhouette = ['Train K-Means','Test K-Means','Train Agglo','Test Agglo']
scores_result_silhouette = [train_sil,test_sil,train_sil_agg,test_sil_agg]
colors = ['red','blue','red','blue']

bars = plt.bar(models_result_silhouette,scores_result_silhouette,color=colors)

for bar,score in zip(bars,scores_result_silhouette):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{score:.4f}', ha='center', va='bottom')

plt.ylabel('Nilai')
plt.xlabel('Kelompok')
plt.title('Hasil visualisasi Train dan Test')
plt.show()