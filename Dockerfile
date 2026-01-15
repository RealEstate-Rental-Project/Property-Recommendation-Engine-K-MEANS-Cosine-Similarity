# Utilisation d'une image Python légère
FROM python:3.10-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code source
COPY . .

# AJOUT CRUCIAL : Indique à Python où trouver vos modules
ENV PYTHONPATH="${PYTHONPATH}:/app/model/app"

# Le port d'écoute configuré dans votre application
EXPOSE 8000

# Commande de lancement
CMD ["python", "model/app/main.py"]
