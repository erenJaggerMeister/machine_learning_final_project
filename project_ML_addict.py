import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

df = pd.read_csv('../Students Social Media Addiction.csv')

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

cdf_2 = df[['Age','Addicted_Score']]
cdf_2 = cdf_2.groupby('Age', as_index=False)['Addicted_Score'].mean()

scaler = StandardScaler()
scaled_data_2 = scaler.fit_transform(cdf_2)

agg = AgglomerativeClustering(n_clusters=3)
cluster_labels_2 = agg.fit_predict(scaled_data_2)
cdf_2['Clusters'] = cluster_labels_2

sil_score = silhouette_score(scaled_data_2, cluster_labels_2)
print(f"Hierarchical Clustering --> Silhouette Score: {sil_score:.4f}")

scatter = plt.scatter(cdf_2['Age'],cdf_2['Addicted_Score'], c=cdf_2['Clusters'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Addicted Score')
plt.title('Skor adiksi terhadap penggunaan sosial media berdasarkan umur')
plt.colorbar(scatter,label='Cluster')
plt.grid(True)
plt.show()