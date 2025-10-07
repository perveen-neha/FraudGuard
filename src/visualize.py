import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(data):
    sns.countplot(data=data, x='Class')
    plt.title('Class Distribution')
    plt.show()

def plot_model_accuracy(df, title):
    sns.barplot(data=df, x='Models', y='ACC', palette='viridis')
    plt.title(title)
    plt.show(block=True)
