import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
ran = np.random.choice(x_norm.index, size = int(0.8 * len(x_norm)), replace = False)

x_train = x_norm.iloc[ran,:]
x_test = x_norm.drop(ran)

y_train = y.iloc[ran]
y_test = y.drop(ran)


# =============================================================================
# Normalisation and Splitting Done
# =============================================================================


log_model = LogisticRegression(max_iter = 1000)
log_model.fit(x_train, y_train)

pred = log_model.predict(x_test)

accuracy = accuracy_score(y_test, pred)*100
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n",conf_matrix)

# =============================================================================
# Accuracy : 86.39
# precision: 0.86
# recall: 1.00
# =============================================================================








