from imblearn.over_sampling import SMOTE
import pandas as pd

def undersample(data):
    normal = data[data['Class']==0]
    fraud = data[data['Class']==1]
    normal_sample = normal.sample(n=len(fraud))
    return pd.concat([normal_sample, fraud], ignore_index=True)

def oversample(X, y):
    return SMOTE().fit_resample(X, y)
