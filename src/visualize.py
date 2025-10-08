import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()

def plot_class_distribution(data,fig):
    if fig:
        plt.figure(fig)
    else:
        plt.figure()
    sns.countplot(data=data, x='Class')
    plt.title('Class Distribution')
    # plt.show(block=True)  

def plot_model_accuracy(df, title,fig):
    if fig:
        plt.figure(fig)
    else:
        plt.figure()
    sns.barplot(data=df, x='Models', y='ACC', palette='viridis')
    plt.title(title)
    # plt.show(block=True)
