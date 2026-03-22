# Projet de Deep Learning VAE

Projet de Deep Learning : Auto-encodeurs Variationnels (VAE)
Ce dépôt a été réalisé dans le cadre du cours "Projet de Deep Learning" du Master 1 "Mathématiques et Intelligence Artificielle" de l'Université Paris-Saclay. Il contient l'implémentation et l'analyse de modèles génératifs profonds, centralisés autour d'une application interactive Streamlit.

Contenu de l'application
L'utilisateur peut naviguer à travers trois onglets principaux pour explorer les modèles :

Théorie : Présentation mathématique des Auto-encodeurs Variationnels. Cette section détaille l'objectif de l'Evidence Lower Bound (ELBO) et le rôle de la divergence KL comme régularisateur de l'espace latent.

Entraînement : Interface dédiée à l'apprentissage des modèles sur les jeux de données MNIST ou Fashion-MNIST.

L'utilisateur peut ajuster les hyperparamètres : Alpha (poids de la reconstruction), Beta (poids de la divergence KL), la dimension latente et le nombre d'époques.

Visualisation en temps réel des courbes de perte (Total, Reconstruction, KL) et des premières reconstructions du modèle.

Gestion intelligente de l'état : les modèles et optimisateurs sont synchronisés avec la configuration choisie dans la barre latérale.

Génération & Espace Latent : Exploration des capacités génératives après entraînement.

Génération Aléatoire (VAE) : Échantillonnage dans le prior Gaussien pour générer de nouvelles images synthétiques.

Génération Conditionnelle (CVAE) : Utilisation des étiquettes (labels) pour diriger la génération vers une classe spécifique (ex: générer uniquement des "Sneakers" ou des "T-shirts").

Manifold de l'espace latent : Visualisation d'une grille 2D de l'espace latent pour observer la continuité et la structure des représentations apprises par l'encodeur.

Aspects Techniques
Inférence Variationnelle
Le projet implémente l'astuce de reparamétrisation (reparameterization trick), permettant au gradient de circuler à travers le nœud stochastique du goulot d'étranglement (bottleneck).

Documentation
Le code source utilise le style de docstrings NumPy. Une documentation complète au format HTML peut être générée via Sphinx pour explorer les détails des classes VAE et CVAE, ainsi que les utilitaires d'entraînement et de visualisation.

### Installation et Lancement

Cloner le projetet se placer à la racine du dossier.
Créer un environnement virtuel (recommandé) :

"""
bash
python -m venv venv
Activer l'environnement virtuel :

Sur Windows :
Bash
.\venv\Scripts\activate

Sur macOS/Linux :

Bash
source venv/bin/activate
Installer les dépendances :

Bash
pip install -r requirements.txt
Lancer l'application :

Bash
streamlit run app.py
"""

## Miscellaneous

### pre-commit usage

This repo uses 2 pre-commit hooks: black and flake8. Contributors should install pre-commit (`pip install pre-commit`) and then run `pre-commit install` to install the hooks. Update the hooks with `pre-commit autoupdate`.

### docstrings

This repo uses the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for its docstrings.