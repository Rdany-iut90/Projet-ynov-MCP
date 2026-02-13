import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature 
from fastmcp import FastMCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Initialisation du serveur MCP
mcp = FastMCP("SimpleModelExpert")

# Configuration MLflow
MLFLOW_TRACKING_URI = "https://mlflow-server-145028569466.europe-west3.run.app/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Test_RandomForest_Simple")

@mcp.tool()
def train_dirty_model(n_trees: int = 10):
    """
    Entraîne un modèle rapide sur des fausses données et log le modèle complet.
    Args:
        n_trees: Nombre d'arbres (plus c'est bas, plus c'est rapide)
    """
    try:
        # 1. Préparation des données
        X, y = make_classification(n_samples=100, n_features=4, n_informative=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 2. Entraînement
        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=3)
        
        with mlflow.start_run() as run:
            print(f"🌲 Entraînement avec {n_trees} arbres...")
            clf.fit(X_train, y_train)

            # 3. Calcul des métriques
            accuracy = clf.score(X_test, y_test)
            
            # 4. Logging des Paramètres et Métriques
            mlflow.log_param("n_trees", n_trees)
            mlflow.log_param("data_source", "synthetic")
            mlflow.log_metric("accuracy", accuracy)

            
            predictions = clf.predict(X_train)
            signature = infer_signature(X_train, predictions)

          
            mlflow.sklearn.log_model(
                sk_model=clf, 
                artifact_path="model_artifact",
                signature=signature,
                input_example=X_train[:5] 
            )
            
            print(f"Terminé. Accuracy: {accuracy:.2f}")

        return (f"Expérience terminée !\n"
                f" Accuracy: {accuracy:.2f}\n"
                f" Run ID: {run.info.run_id}\n"
                f" Le modèle a été sauvegardé dans l'artefact 'model_artifact'.")

    except Exception as e:
        return f"Erreur : {str(e)}"
    
@mcp.tool()
def register_model_to_registry(run_id: str, model_name: str, artifact_path: str = "model_artifact"):
    """
    Enregistre un modèle depuis une Run existante vers le Model Registry de MLflow.
    
    Args:
        run_id: L'identifiant de la run (récupéré lors de l'entraînement).
        model_name: Le nom unique à donner au modèle dans le registre (ex: "RandomForest_Production").
        artifact_path: Le dossier où le modèle est stocké (par défaut "model_artifact" comme dans train_dirty_model).
    """
    try:
        print(f" Tentative d'enregistrement du run {run_id} sous le nom {model_name}...")
        

        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        return (f" Succès ! Modèle enregistré.\n"
                f"Nom : {registered_model.name}\n"
                f"Version : {registered_model.version}\n"
                f"Source : {model_uri}")

    except Exception as e:
        return f"❌ Erreur lors de l'enregistrement : {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run("sse", host="0.0.0.0", port=port)