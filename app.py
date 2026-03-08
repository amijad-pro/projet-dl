"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
import streamlit as st  
import torch
import torch.optim as optim
from dl import VAE, train_model
from utils import get_mnist_loaders, get_frey_loader
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. CONFIGURATION INITIALE
st.set_page_config(page_title="VAE Project - Dashboard")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. BARRE LATÉRALE (Paramètres)
st.sidebar.header("Configuration du Modèle")
dataset_name = st.sidebar.selectbox("Choisir le Dataset", ["MNIST", "Frey Face"])
alpha = st.sidebar.slider("Alpha (Reconstruction)", 0.0, 2.0, 1.0)
beta = st.sidebar.slider("Beta (Régularisation KL)", 0.0, 5.0, 1.0)
epochs = st.sidebar.slider("Nombre d'époques", 1, 20, 3)

# 3. CHARGEMENT DES DONNÉES ET DU MODÈLE
# Ces variables doivent exister AVANT que l'on clique sur le bouton
if dataset_name == "MNIST":
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    input_dim = 784
else:
    # Assure-toi que le fichier .mat est bien dans le dossier /data
    train_loader = get_frey_loader(batch_size=64)
    input_dim = 560 # 28x20

model = VAE(input_dim=input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. FONCTION DE VISUALISATION
def plot_reconstructions(model, dataset, n=10):
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    
    with torch.no_grad():
        for i in range(n):
            img, _ = dataset[i]
            # On aplatit l'image pour le modèle
            img_input = img.view(-1, input_dim).to(device)
            recon, _, _ = model(img_input)
            
            # Image Originale
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].axis('off')
            
            # Image Reconstruite
            shape = (28, 28) if dataset_name == "MNIST" else (28, 20)
            axes[1, i].imshow(recon.view(shape).cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
    
    st.pyplot(fig)

# 5. INTERFACE PRINCIPALE
st.title("Analyse du VAE - Dynamique Stochastique")
st.write(f"Dataset sélectionné : **{dataset_name}**")


if st.button("Lancer l'entraînement"):
    # init historiques de perte 
    history = {"Total": [], "Reconstruction": [], "Régularisation (KL)": []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    image_container = st.container() 

    for epoch in range(1, epochs + 1):
        # On récupère le dictionnaire des pertes (on passe les paramètres alpha et beta à la fonction d'entraînement)
        losses = train_model(model, train_loader, optimizer, epoch, alpha, beta)
        
        # Sauvegarde pour le graphique
        history["Total"].append(losses["total"])
        history["Reconstruction"].append(losses["bce"])
        history["Régularisation (KL)"].append(losses["kld"])
        
        progress_bar.progress(epoch / epochs)
        status_text.text(f"Époque {epoch} - Perte Totale: {losses['total']:.2f}")
        
        with image_container:
            st.write(f"### Visualisation Époque {epoch}")
            plot_reconstructions(model, train_loader.dataset)
            
    st.success("Entraînement terminé !")

    # AFFICHAGE DES GRAPHIQUES 
    st.divider()
    st.header("Analyse des Pertes (Loss Analysis)")
    
    # Création d'une figure Matplotlib avec 3 sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(history["Total"], color='blue')
    ax1.set_title("Perte Totale")
    ax1.grid(True)
    
    ax2.plot(history["Reconstruction"], color='green')
    ax2.set_title("Reconstruction (BCE)")
    ax2.grid(True)
    
    ax3.plot(history["Régularisation (KL)"], color='red')
    ax3.set_title("Régularisation (KLD)")
    ax3.grid(True)
    
    st.pyplot(fig)

st.divider() # Crée une ligne de séparation visuelle
st.header("Exploration de l'Espace Latent (Manifold)")

def plot_latent_space(model, dataset_name, n=15, digit_size=28):
    """Génère une grille d'images en parcourant l'espace latent."""
    model.eval()
    # Ajustement de la taille selon le dataset
    height, width = (28, 28) if dataset_name == "MNIST" else (28, 20)
    figure = np.zeros((height * n, width * n))
    
    # Grille de coordonnées latentes (de -3 à 3 écart-types)
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # On crée un point z=(xi, yi) et on met le reste à 0
            z_sample = torch.zeros(1, model.fc2_mu.out_features).to(device)
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi
            
            with torch.no_grad():
                sample = model.decode(z_sample).cpu().numpy()
                digit = sample.reshape(height, width)
                figure[i * height: (i + 1) * height,
                       j * width: (j + 1) * width] = digit

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(figure, cmap='Greys_r')
    ax.axis('off')
    st.pyplot(fig)

# Ajout d'une case à cocher pour ne pas ralentir l'app au démarrage
if st.checkbox("Afficher la grille de génération (Manifold)"):
    st.write("Visualisation des transitions fluides dans l'espace latent $z_1$ et $z_2$.")
    plot_latent_space(model, dataset_name)