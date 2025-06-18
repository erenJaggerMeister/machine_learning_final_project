import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../Students Social Media Addiction.csv')

# print(df.isnull().sum())

# print(df.head())

# print(df.describe())

# cdf = df[['Age','Academic_Level','Avg_Daily_Usage_Hours','Most_Used_Platform','Affects_Academic_Performance','Addicted_Score']]

# print(cdf.head(9))

# viz = cdf[['Age','Academic_Level','Avg_Daily_Usage_Hours','Most_Used_Platform','Affects_Academic_Performance','Addicted_Score']]
# viz.hist()
# plt.show()

# plt.scatter(cdf.Age, cdf.Avg_Daily_Usage_Hours)
# plt.xlabel("Age")
# plt.ylabel("Avg Usage Hours")
# plt.show()

# plt.bar(cdf.Age, cdf.Avg_Daily_Usage_Hours)
# plt.xlabel("Age")
# plt.ylabel("Avg Usage Hours")
# plt.show()

# cdf = df[['Age','Avg_Daily_Usage_Hours','Addicted_Score']]

# grouped = cdf.groupby('Age').mean().reset_index()

# print(grouped)
# plt.plot(grouped['Age'], grouped['Addicted_Score'], marker='o', label='Rata-rata Skor Adiksi')
# plt.plot(grouped['Age'], grouped['Avg_Daily_Usage_Hours'], marker='o', label='Rata-rata Jam Pemakaian')
# plt.title("Pengaruh Rata-rata Penggunaan Media Sosial terhadap Skor Adiksi Berdasarkan Umur")
# plt.xlabel("Umur")
# plt.ylabel("Nilai")
# plt.legend()
# plt.grid(True)
# plt.show()

cdf = df[['Age', 'Avg_Daily_Usage_Hours', 'Addicted_Score']].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cdf)

kmeans = KMeans(n_clusters=3, random_state=42)
cdf['Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10,6))
scatter = plt.scatter(cdf['Age'], cdf['Avg_Daily_Usage_Hours'], c=cdf['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Avg Daily Usage Hours')
plt.title('Clustering Pengguna Berdasarkan Usia dan Rata-rata Penggunaan')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()