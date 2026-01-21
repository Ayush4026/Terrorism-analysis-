import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data = pd.read_excel("/Users/chauh/Downloads/project_dataset.xlsx")

print(data.shape)
print(data.info())

req_columns = ['iyear', 'imonth', 'country', 
               'region', 'attacktype1','targtype1',
               'weaptype1','nkill','nwound',
               'suicide']

df = data[req_columns]

print(df.head())

df.isnull().sum()

# =============================================================================
# Null values-
#     nkill = 424
#     nwound = 769
# =============================================================================

df['nkill'] = df['nkill'].fillna(0)
df['nwound'] = df['nwound'].fillna(0)

df.isnull().sum()

(df[['nkill', 'nwound']] < 0).sum()

print(df.head())
df.info()
df.describe()

# =============================================================================
# Cleaning Done 
# =============================================================================

scaler = MinMaxScaler()
df_norm = scaler.fit_transform(df)

wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters = k , random_state= 42)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss, marker = 'o')
plt.title("Elbow method for optimal K")
plt.xlabel("Number of clusters (k)")
plt.ylabel("wcss")
plt.grid(True)
plt.plot()

print("wcss values for each k = 1 to 10:")
for i, val in enumerate(wcss , 1):
    print(f"k = {i} -> wcss = {val}")

print("By analysing the graph , the optimal no. of K is: 3")













