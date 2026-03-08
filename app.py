"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
<<<<<<< HEAD
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
=======

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from viz import mnist_like_viz, training_curves
from utils import poly, paths
import dl


def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar. The mode function that can be called are
    `regression`, `sinus`, `mnist_viz`, and `fashionmnist`.

    Returns
    -------
    None
    """
    st.title("Some data manipulations")

    home_data = get_data()

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Show instructions",
            "Home data regression",
            "Sinus regression",
            "Show MNIST",
            "Deep Learning",
        ],
    )  # , "Show the source code"])
    if app_mode == "Show instructions":
        st.write("To continue select a mode in the selection box to the left.")
    # elif app_mode == "Show the source code":
    #     st.code(get_file_content_as_string("./app.py"))
    elif app_mode == "Home data regression":
        regression(home_data)
    elif app_mode == "Sinus regression":
        sinus()
    elif app_mode == "Show MNIST":
        mnist()
    elif app_mode == "Deep Learning":
        fashionmnist()


@st.cache_data
def get_data():
    """Loads the home training data.

    Returns
    -------
    home_data: pd.DataFrame
        The home training data.

    Notes
    -----
    This is the dataset dowloaded from https://www.kaggle.com/competitions/home-data-for-ml-course/data.

    """
    iowa_file_path = "./home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)
    return home_data


# def get_file_content_as_string(path):
#     with open(path) as f:
#         lines = f.read()
#     return lines


def regression(home_data):
    """Performs regression on the home training data.

    The dataset is split in a training and
    a validation sets.
    The user has the choice of which covariates to incoporate
    in the model. Then a decision tree, a decision tree
    with `max_leaf_nodes=100`, and a random forest are fitted
    on the training set. Finally the validation mean
    absolute errors are displayed.

    Parameters
    ----------
    home_data: pd.DataFrame
        The home training data. It can be any DataFrame except it needs
        the columns `SalePrice`, `LotArea`, `YearBuilt`, `1stFlrSF`,
        `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, and `TotRmsAbvGrd`.

    Returns
    -------
    None

    """
    # Create target object and call it y
    y = home_data.SalePrice

    features = [
        "LotArea",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
    ]
    home_data_extracted = home_data[["SalePrice"] + features]

    st.text(
        "This is the head of the dataframe of Iowa house prices with many covariates"
    )
    st.write(home_data_extracted.head())

    # Create X
    covariates = st.multiselect(
        "Select covariates to keep for regression:", features, features
    )
    covariates.sort()
    X = home_data[covariates]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    dict_val_maes = {"method": [], "Val MAE": []}

    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)
    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor")
    dict_val_maes["Val MAE"].append(val_mae)

    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor with max leaf nodes")
    dict_val_maes["Val MAE"].append(val_mae)

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    dict_val_maes["method"].append("RandomForestRegressor")
    dict_val_maes["Val MAE"].append(rf_val_mae)

    val_maes = pd.DataFrame(dict_val_maes).set_index("method")
    st.write(val_maes)
    st.text("(Test what happens when removing TotRmsAbvGrd)")


def sinus():
    """A simple example of regression on the sinus function on the interval [0,5].

    Some points are perturbed with noise after applying
    the sinus function to them.
    The user decides the number of noisy points with a slider,
    and the maximum order for the polynomial regression. They
    also decide if they want to fit two regression trees (with
    `max_depth=2` and `max_depth=5`) in addition to the polynomial
    regression. Then the fitted models are plotted along with
    the training noisy data.

    Returns
    -------
    None

    """
    noise = st.slider("Noise volume", 1, 10, 5, format="1 of each %d point(s)")
    # Order of the polynom for the linear regression with polynom
    order = st.slider(
        "Choose the order of the polynom for the polynomial regression", 2, 20, 3
    )
    trees = st.checkbox("Show decision trees", True)

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1))
    y = np.sin(X).ravel()
    y[::noise] += 3 * (0.5 - rng.rand(y[::noise].size))
    X2 = poly(X, order=order)

    # Fit regression models
    if trees:
        regr_1 = DecisionTreeRegressor(max_depth=2, random_state=1)
        regr_2 = DecisionTreeRegressor(max_depth=5, random_state=1)
    regr_3 = LinearRegression()
    if trees:
        regr_1.fit(X, y)
        regr_2.fit(X, y)
    regr_3.fit(X2, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    X2_test = poly(X_test, order=order)
    if trees:
        y_1 = regr_1.predict(X_test)
        y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X2_test)

    # Plot the results
    fig = plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    if trees:
        plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
        plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.plot(X_test, y_3, color="red", label="polynom", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    if trees:
        plt.title("Decision Trees and Polynomial Regression")
    else:
        plt.title("Polynomial Regression")
    plt.xlim(-0.2, 5.2)
    plt.ylim(-2.7, 2.7)
    plt.legend()
    st.pyplot(fig)


def mnist():
    """Selects randomly 6 images from the training MNIST dataset and displays them.

    Returns
    -------
    None

    """
    train_data = MNIST("data", train=True, download=True, transform=ToTensor())
    classes = list(range(10))
    mnist_like_viz(train_data, classes)


def fashionmnist():
    """Training a simple MLP on the FashionMNIST dataset and displaying the metrics evolution during the training.

    The user can decide the number of hidden layers of the MLP. They can also choose the number of epochs
    for training. Once a model with given hyperparameters is trained, it is saved and used
    again the next times without new training, unless the user clicks the button to delete
    the saved model and train again. The MLP architecture is displayed.
    Then 2 figures that are the evolution
    of, respectively, the losses (train and test) and accuracies (train and test)
    with respect to the epoch, are displayed. Finally 6 random images of the test dataset are
    displayed, along with their ground truth and predicted labels.

    Returns
    -------
    None

    Notes
    -----
    Inspired by https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.

    """
    st.header("A simple deep learning model applied on the FashionMNIST dataset")

    hidden_layers = st.slider("Choose the number of hidden layers", 1, 5, 2)

    dropout_rate = st.slider("Choose the dropout rate", 0.0, 0.9, 0.0, 0.1)

    epochs = st.slider("Choose the number of epochs to train", 1, 1000, 50)
    st.write(
        "Note that the epoch parameter is only relevant for training a new model, so if there is no already saved model for this config"
    )

    if st.button("Delete saved model and train again"):
        path_weights, path_metrics = paths(hidden_layers, dropout_rate)
        try:
            os.remove(path_weights)
            os.remove(path_metrics)
        except FileNotFoundError:
            pass

    train_dataloader, test_dataloader, _, test_data = dl.get_FashionMNIST_datasets(
        64, only_loader=False
    )
    model = dl.get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        epochs=epochs,
        mode="st",
    )

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    training_curves(model, "st")
    mnist_like_viz(test_data, classes, model)


if __name__ == "__main__":
    main()
>>>>>>> 266a5492f9f3d75528fb99e658271c452215da96
