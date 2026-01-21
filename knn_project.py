import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
data = pd.read_excel("/Users/chauh/Desktop/ML_project/project_dataset.xlsx")

print(data.shape)
print(data.info())

req_columns = ['iyear', 'imonth', 'country', 
               'region', 'attacktype1','targtype1',
               'weaptype1','nkill','nwound',
               'suicide','success']

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

df.info()
df.describe()

# =============================================================================
# Cleaning Done 
# =============================================================================

x = df.drop('success' , axis = 1)
y = df['success']

scaler = MinMaxScaler()

x_norm = pd.DataFrame(scaler.fit_transform(x),columns= x.columns)

np.random.seed(42)
ran = np.random.choice(x_norm.index, size = int(0.9 * len(x_norm)), replace = False)

x_train = x_norm.iloc[ran,:]
x_test = x_norm.drop(ran)

y_train = y.iloc[ran]
y_test = y.drop(ran)

# =============================================================================
# Normalisation and Splitting Done
# =============================================================================

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)

tab = confusion_matrix(y_test, pred)
print("Confusion Matrix: \n",tab )

acc = accuracy_score(y_test, pred) * 100
print("Accuracy Score: {:.2f}%".format(acc))

# =============================================================================
# Accuracy Score = 86.69%
# =============================================================================



plt.figure(figsize=(7,5))
plt.hist(df['nkill'], bins=30)
plt.xlabel("Number of Kills")
plt.ylabel("Frequency")
plt.title("Histogram of Number of Kills")
plt.grid(True)
plt.show()






















