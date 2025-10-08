from src.preprocess import load_and_preprocess
from src.sampling import undersample, oversample
from src.models import train_models
from src.evaluate import evaluate_model          
from src.visualize import plot_class_distribution, plot_model_accuracy
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



data = load_and_preprocess('data/creditcard.csv')
plot_class_distribution(data,1)
plt.show(block=False)

x=data.drop('Class', axis=1)
y=data['Class']

data_uder = undersample(data)
x_train, x_test, y_train, y_test = train_test_split(data_uder.drop('Class', axis=1), data_uder['Class'], test_size=0.2, random_state=42)

models = train_models(x_train, y_train)
results = {name: evaluate_model(model, x_test, y_test) for name, model in models.items()}

final_data = pd.DataFrame({
    "Models" : results.keys(),
       "ACC" : [r["Accuracy"]*100 for r in results.values()]
})

plot_model_accuracy(final_data, 'Model Accuracy on Undersampled Data',2)       
plt.show(block=False)
# Oversampling

X_res, y_res = oversample(x, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
models = train_models(X_train, y_train)
results = {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}
final_data = pd.DataFrame({
    "Models": results.keys(),
    "ACC": [r["Accuracy"] * 100 for r in results.values()]
})
plot_model_accuracy(final_data, "Model Accuracy Comparison (Oversampling)",3)
plt.show(block=False)

input("Press Enter to exit...")