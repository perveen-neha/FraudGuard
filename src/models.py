from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {
        "LR": LogisticRegression(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
