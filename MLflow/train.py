# train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


mlflow.set_tracking_uri("https://votre-mlflow-sur-azure.azurewebsites.net") 
mlflow.set_experiment("Projet_Groupe_MCP")

def train():
    # 1. Chargement des données
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    with mlflow.start_run():
      
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)

        # 3. Evaluation
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        
       
        mlflow.sklearn.log_model(clf, "model_iris")
        print(f"Modèle entraîné et envoyé sur Azure. Acc: {acc}")

if __name__ == "__main__":
    train()