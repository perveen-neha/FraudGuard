import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates()
    data['Amount'] = StandardScaler().fit_transform(data[['Amount']])
    data = data.drop(['Time'], axis=1)
    return data