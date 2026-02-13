#!/bin/bash
set -e

# --- VARIABLES ---
PROJECT_ID="ynov-raph"
APP_NAME="mlflow-server"
REGION="europe-west3"
BUCKET_NAME="mlflow-artifacts-${PROJECT_ID}"

echo "🚀 DÉBUT DU DÉPLOIEMENT (Mode sans gsutil)"

# 1. Configuration
gcloud config set project "$PROJECT_ID"

# 2. Déploiement Direct (On suppose que le bucket existe déjà)
echo "🚀 Déploiement sur Cloud Run..."

gcloud run deploy "$APP_NAME" \
  --source . \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port $PORT \
  --set-env-vars ARTIFACT_ROOT="gs://$BUCKET_NAME"

echo "✅ FINI ! Vérifie l'URL ci-dessus."

