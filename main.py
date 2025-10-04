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

try:
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

    sns.countplot(data['Class'])
    plt.show()
    
    print("--- Script finished ---")
    input("Press Enter to exit...")

except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(e)
    input("Press Enter to exit...")