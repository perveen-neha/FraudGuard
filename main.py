import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE



print("--- Script starting ---")

# loading the dataset to a Pandas DataFrame
print("1. Loading data from 'data/creditcard.csv'...")
data = pd.read_csv('data/creditcard.csv')
print("...Data loaded successfully. Shape:", data.shape)

# first 5 rows of the dataset
print("2. Checking for initial duplicates...")
print(data.duplicated().any())
data = data.drop_duplicates()
print("...Duplicates dropped. New shape:", data.shape)

print("3. Scaling the 'Amount' column...")
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
print("...'Amount' column scaled.")

print("4. Dropping the 'Time' column...")
data = data.drop(['Time'], axis=1)
print("...'Time' column dropped.")

print("5. Checking final shape and class counts...")
print("Final shape:", data.shape)
print(data['Class'].value_counts())

print("--- Preprocessing complete. About to plot. ---")

# This input should now be reached
# input("Press Enter before plotting...")

sns.countplot( data=data, x='Class')
plt.title('Class Distribution')
# plt.show()

X = data.drop('Class',axis=1)
y = data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)

# 9. Handling Imbalanced Dataset¶
# Undersampling

normal = data[data['Class']==0]
fraud = data[data['Class']==1]
normal.shape
fraud.shape
normal_sample=normal.sample(n=473)
normal_sample.shape

new_data = pd.concat([normal_sample,fraud],ignore_index=True)
new_data['Class'].value_counts()
new_data.head()
X = new_data.drop('Class',axis=1)
y = new_data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)

log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2 = dt.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)
print(accuracy_score(y_test,y_pred3))   
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})
print(final_data)

# Passing the whole DataFrame and referencing columns by name
plt.figure()
sns.barplot(data=final_data, x='Models', y='ACC', palette='viridis')
plt.title('Model Accuracy Comparison')
plt.show()

# oversampling
print("6. Applying SMOTE for oversampling...")
X = data.drop('Class',axis=1)
y = data['Class']

X_res,y_res = SMOTE().fit_resample(X,y)
y_res.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20, random_state=42)

# Logistic Regression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2 = dt.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))      
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))

rf = RandomForestClassifier(
     n_estimators=30,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train,y_train) 
y_pred3 = rf.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3)) 
print(f1_score(y_test,y_pred3))

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})
print(final_data)

plt.figure()
sns.barplot(data=final_data, x='Models', y='ACC', palette='viridis')
plt.title('Model Accuracy Comparison after Oversampling')
plt.show(block=True)  