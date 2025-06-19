import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../Students Social Media Addiction.csv')

cdf = df[['Age', 'Avg_Daily_Usage_Hours', 'Addicted_Score']].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cdf)

dbscan = DBSCAN(eps=0.8, min_samples=5)
cluster_labels = dbscan.fit_predict(scaled_data)
cdf['Cluster'] = cluster_labels

plt.figure(figsize=(10,6))
scatter = plt.scatter(cdf['Age'], cdf['Avg_Daily_Usage_Hours'], c=cdf['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Avg Daily Usage Hours')
plt.title('Clustering Pengguna Berdasarkan Usia dan Rata-rata Penggunaan')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()