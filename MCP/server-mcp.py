import os
import mlflow
import mlflow.sklearn
from fastmcp import FastMCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Initialisation du serveur MCP
mcp = FastMCP("SimpleModelExpert")

MLFLOW_TRACKING_URI = "https://mlflow-server-145028569466.europe-west3.run.app/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Test_RandomForest_Simple")

@mcp.tool()
def train_dirty_model(n_trees: int = 10):
    """
    Entraîne un modèle rapide sur des fausses données et log le résultat.
    Args:
        n_trees: Nombre d'arbres (plus c'est bas, plus c'est rapide/pourri)
    """
    try:
        X, y = make_classification(n_samples=100, n_features=4, n_informative=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=3)

     
        with mlflow.start_run() as run:
            print(f" Entraînement avec {n_trees} arbres...")
            clf.fit(X_train, y_train)

            # Calculer la précision
            accuracy = clf.score(X_test, y_test)
            
            # Logger les paramètres et les métriques
            mlflow.log_param("n_trees", n_trees)
            mlflow.log_param("data_source", "synthetic")
            mlflow.log_metric("accuracy", accuracy)

            # Logger le modèle (format sklearn)
            mlflow.sklearn.log_model(clf, "random_forest_model")
            
            print(f"✅ Terminé. Accuracy: {accuracy:.2f}")

        return (f"Expérience terminée !\n"
                f" Accuracy: {accuracy:.2f}\n"
                f" Run ID: {run.info.run_id}\n"
                f"Tu peux voir le résultat sur ton serveur MLflow.")

    except Exception as e:
        return f"❌ Erreur : {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run("sse", host="0.0.0.0", port=port)