import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from hmc import bma_inference, set_model_params

# Fonction pour mettre à jour les paramètres du modèle
#def set_model_params(model, theta):
#    with torch.no_grad():
#        for param, new_param in zip(model.parameters(), theta):
#            param.copy_(new_param)


def evaluate_model(model, test_loader, samples, device="cuda", n_classes=10, verbose=False):
    """
    Évalue les performances du modèle sur un ensemble de test en utilisant BMA et calcule les courbes de calibration.

    Paramètres :
        model (torch.nn.Module) : Le modèle PyTorch.
        test_loader (DataLoader) : DataLoader pour le jeu de données de test.
        samples (torch.Tensor) : Échantillons de poids HMC de forme (M, N, D).
        device (str) : "cuda" ou "cpu" (par défaut "cuda").
        n_classes (int) : Le nombre de classes (par défaut 10).
    
    Retourne :
        dict : Les résultats d'évaluation, y compris l'accuracy, la confiance et l'ECE.
    """
    
    # Liste pour stocker les sorties et les cibles
    outputs = []
    targets = []
    
    # Mode évaluation
    model.eval()

    # Désactivation du calcul des gradients
    with torch.no_grad():
        # Boucle sur les données de test
        for data, target in tqdm(test_loader, desc="Predicting the test set"):
            data, target = data.to(device), target.to(device)  # Envoi des données sur le GPU
            
            # Obtenir les prédictions avec BMA
            preds, avg_pred = bma_inference(model, samples, data, device)
            
            # Ajouter les prédictions et cibles à la liste
            outputs.append(preds)
            targets.append(target)
    
    # Conversion des listes en tenseurs
    outputs = torch.stack(outputs).view(-1, n_classes)
    targets = torch.stack(targets).view(-1)
    
    # Calcul de la courbe de calibration
    results = compute_calibration_curve(outputs, targets)
    to_return ={}
    to_return['accuracy'] = results["accuracy"].mean().item()
    to_return['confidence'] = results["confidence"].mean().item()
    to_return['ECE'] = results['ece']
    to_return['log_likelihood'] = -F.cross_entropy(outputs, targets, reduction='sum').item()

    if verbose:
        # Affichage des résultats
        print("\nAccuracy : ", to_return['accuracy'] )
        print("Confidence : ", to_return['confidence'])
        print("ECE : ", to_return['ECE'])
        print("log likelihood : ", to_return['log_likelihood'])

    return to_return

def evaluate_single_model(model, test_loader, sample, device="cuda", n_classes=10, verbose=False):
    """
    Évalue les performances d'un modèle spécifique avec un seul échantillon de poids.

    Paramètres :
        model (torch.nn.Module) : Le modèle PyTorch.
        test_loader (DataLoader) : DataLoader pour le jeu de données de test.
        sample (list[torch.Tensor]) : Un échantillon de poids HMC (une seule réalisation).
        device (str) : "cuda" ou "cpu" (par défaut "cuda").
        n_classes (int) : Le nombre de classes (par défaut 10).
    
    Retourne :
        dict : Les résultats d'évaluation, y compris l'accuracy, la confiance et l'ECE.
    """
    
    # Appliquer l'échantillon de poids au modèle
    set_model_params(model, sample)  

    # Liste pour stocker les sorties et les cibles
    outputs = []
    targets = []
    
    # Mode évaluation
    model.eval()

    # Désactivation du calcul des gradients
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Predicting the test set"):
            data, target = data.to(device), target.to(device)  
            
            # Prédictions avec un seul ensemble de poids
            logits = model(data)
            preds = torch.softmax(logits, dim=1)  

            # Ajouter les prédictions et cibles à la liste
            outputs.append(preds)
            targets.append(target)
    
    # Conversion des listes en tenseurs
    outputs = torch.cat(outputs, dim=0)  
    targets = torch.cat(targets, dim=0)

    # Calcul de la courbe de calibration
    results = compute_calibration_curve(outputs, targets)
    
    to_return = {
        'accuracy': results["accuracy"].mean().item(),
        'confidence': results["confidence"].mean().item(),
        'ECE': results['ece'],
        'log_likelihood': -F.cross_entropy(outputs, targets, reduction='sum').item()
    }

    if verbose:
        print("\nAccuracy : ", to_return['accuracy'])
        print("Confidence : ", to_return['confidence'])
        print("ECE : ", to_return['ECE'])
        print("log likelihood : ", to_return['log_likelihood'])

    return to_return


def compute_calibration_curve(outputs, labels, num_bins=20):
    """
    Compute the calibration curve and Expected Calibration Error (ECE).

    Parameters:
        outputs (torch.Tensor): Model outputs (probabilities), shape (N, num_classes).
        labels (torch.Tensor): Ground-truth labels, shape (N,).
        num_bins (int): Number of bins for calibration.

    Returns:
        dict: Contains "confidence", "accuracy", "proportions", and "ece".
    """
    # Flatten outputs and labels
    labels = labels.view(-1)  # Ensure labels are 1D
    outputs = outputs.view(labels.size(0), -1)  # Ensure correct shape

    # Get confidences (max probabilities) and predicted classes
    confidences, predictions = torch.max(outputs, dim=1)
    accuracies = (predictions == labels).float()  # Boolean to float (1=correct, 0=incorrect)

    # Sort confidences to determine bin edges
    num_inputs = confidences.shape[0]
    bins = torch.sort(confidences)[0][::(num_inputs + num_bins - 1) // num_bins]
    if num_inputs % ((num_inputs + num_bins - 1) // num_bins) != 1:
        bins = torch.cat((bins, confidences.max().unsqueeze(0)))

    # Initialize bin statistics
    bin_confidences, bin_accuracies, bin_proportions = [], [], []
    ece = 0.0  # Expected Calibration Error

    # Compute calibration metrics for each bin
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_proportions.append(prop_in_bin)

    # Convert lists to tensors
    return {
        "confidence": torch.tensor(bin_confidences),
        "accuracy": torch.tensor(bin_accuracies),
        "proportions": torch.tensor(bin_proportions),
        "ece": ece
    }


def compute_dataset_log_prob(model, dataloader, prior_std=1.0, device="cuda"):
    """
    Calcule le log prior, log likelihood et log posterior sur tout le dataset.

    Parameters:
        model (torch.nn.Module): Le modèle PyTorch.
        dataloader (torch.utils.data.DataLoader): Le DataLoader du dataset.
        prior_std (float): Écart-type du prior gaussien.
        device (str): "cuda" ou "cpu" selon le matériel.

    Returns:
        dict: Contenant "log_prior", "log_likelihood" et "log_posterior".
    """
    model.eval()  # Met le modèle en mode évaluation (important pour désactiver dropout, batchnorm...)
    log_likelihood_total = 0.0
    total_samples = 0

    # Calcul du log prior (indépendant des données)
    log_prior = -0.5 * sum(torch.sum(p ** 2) for p in model.parameters()) / (prior_std ** 2)

    # Calcul du log likelihood en parcourant tout le dataset
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)  # Déplacement sur le bon device

            logits = model(data)
            log_likelihood = -F.cross_entropy(logits, target, reduction='sum')

            log_likelihood_total += log_likelihood.item()
            total_samples += target.size(0)

    # Normalisation par le nombre d'échantillons pour obtenir une moyenne
    log_likelihood_mean = log_likelihood_total / total_samples
    log_posterior = log_likelihood_mean + log_prior  # Posterior = Prior + Likelihood

    return {
        "log_prior": log_prior.item(),
        "log_likelihood": log_likelihood_mean,
        "log_posterior": log_posterior.item()
    }


def visualize_posterior_density(model, test_loader, samples, prior_std=1.0, device="cuda", grid_size=15):
    """
    Visualisation de la densité du posterior log-density, log-likelihood et log-prior dans 
    un sous-espace affine de paramètres défini par trois échantillons HMC.

    Parameters:
        model (torch.nn.Module): Modèle PyTorch
        test_loader (torch.utils.data.DataLoader): Dataloader pour évaluer les métriques
        samples (list of torch.Tensor): Liste contenant au moins 102 poids échantillonnés via HMC
        prior_std (float): Écart-type du prior
        device (str): "cuda" ou "cpu"
        grid_size (int): Nombre de points dans chaque dimension pour la visualisation
    """
    # Sélection des trois échantillons de poids définissant l’espace affine
    tier = int(len(samples) / 3)
    w1, w2, w3 = samples[0], samples[10], samples[60]

    # Génération des coefficients a et b dans [0,1]
    a_vals = np.linspace(-1, 2, grid_size)
    b_vals = np.linspace(-1, 2, grid_size)
    
    # Matrices pour stocker les valeurs des métriques
    log_priors = np.zeros((grid_size, grid_size))
    log_likelihoods = np.zeros((grid_size, grid_size))
    log_posteriors = np.zeros((grid_size, grid_size))

    # Parcours de la grille 2D
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            # Interpolation des poids selon la formule donnée
            w_interp = [(w1_k * a + w2_k * b + w3_k * (1 - a - b)) for w1_k, w2_k, w3_k in zip(w1, w2, w3)]
            
            # Appliquer les poids interpolés au modèle
            set_model_params(model, w_interp)
            
            # Calculer les métriques
            metrics = compute_dataset_log_prob(model, test_loader, prior_std=prior_std, device=device)
            
            # Stocker les valeurs
            log_priors[i, j] = metrics["log_prior"]
            log_likelihoods[i, j] = metrics["log_likelihood"]
            log_posteriors[i, j] = metrics["log_posterior"]

    # Création des figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Création des grilles pour affichage
    A, B = np.meshgrid(a_vals, b_vals)

    # Log Prior
    ax = axes[0]
    contour = ax.contourf(A, B, log_priors.T, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_title("Log Prior")
    ax.set_xlabel("a")
    ax.set_ylabel("b")

    # Ajouter des points distinctifs : (0, 0), (1, 0) et (0, 1)
    ax.plot(0, 0, 'k.', markersize=5)  # Point noir pour (0, 0)
    ax.plot(1, 0, 'k.', markersize=5)  # Point noir pour (1, 0)
    ax.plot(0, 1, 'k.', markersize=5)  # Point noir pour (0, 1)

    # Log Likelihood
    ax = axes[1]
    contour = ax.contourf(A, B, log_likelihoods.T, cmap="plasma")
    fig.colorbar(contour, ax=ax)
    ax.set_title("Log Likelihood")
    ax.set_xlabel("a")
    ax.set_ylabel("b")

    # Ajouter des points distinctifs : (0, 0), (1, 0) et (0, 1)
    ax.plot(0, 0, 'k.', markersize=5)  # Point noir pour (0, 0)
    ax.plot(1, 0, 'k.', markersize=5)  # Point noir pour (1, 0)
    ax.plot(0, 1, 'k.', markersize=5)  # Point noir pour (0, 1)

    # Log Posterior
    ax = axes[2]
    contour = ax.contourf(A, B, log_posteriors.T, cmap="inferno")
    fig.colorbar(contour, ax=ax)
    ax.set_title("Log Posterior")
    ax.set_xlabel("a")
    ax.set_ylabel("b")

    # Ajouter des points distinctifs : (0, 0), (1, 0) et (0, 1)
    ax.plot(0, 0, 'k.', markersize=5)  # Point noir pour (0, 0)
    ax.plot(1, 0, 'k.', markersize=5)  # Point noir pour (1, 0)
    ax.plot(0, 1, 'k.', markersize=5)  # Point noir pour (0, 1)

    plt.tight_layout()
    plt.show()


def compute_r_hat(chains):
    """
    Calcule la statistique R-hat de Gelman & Rubin pour un ensemble de chaînes.

    Paramètres :
        chains (torch.Tensor) : tenseur de forme (M, N, D) avec M chaînes, N échantillons par chaîne et D dimensions.
    
    Retourne :
        torch.Tensor : valeurs de R-hat pour chaque dimension.
    """
    M, N, D = chains.shape  # Nombre de chaînes, nombre d'échantillons, nombre de dimensions

    # Moyenne intra-chaîne de chaque paramètre
    psi_moyenne = chains.mean(dim=1)  # (M, D)

    # Moyenne globale de chaque paramètre
    psi_global = psi_moyenne.mean(dim=0)  # (D,)

    # Variance inter-chaîne B de chaque paramètre
    B = (N / (M - 1)) * ((psi_moyenne - psi_global) ** 2).sum(dim=0)  # (D,)

    # Variance intra-chaîne W de chaque paramètre
    W = (1 / (M * (N - 1))) * ((chains - psi_moyenne[:, None, :]) ** 2).sum(dim=(0, 1))  # (D,)

    # Estimation ajustée de la variance totale de chque paramètre
    sigma_plus = ((N - 1) / N) * W + (B / N)

    # Calcul final de R-hat
    R_hat = ((M + 1) / M) * (sigma_plus / W) - (N - 1) / (M * N) # (D,)

    return R_hat

def plot_r_hat_histogram(r_hat_values, title):
    """
    Affiche un histogramme des valeurs de R-hat.

    Paramètres :
        r_hat_values (torch.Tensor) : Valeurs de R-hat pour chaque paramètre.
        title (str): Titre du graphique.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(r_hat_values.cpu().numpy(), bins=50, log=True, color='b', alpha=0.7)
    plt.axvline(1.1, color='r', linestyle='dashed', linewidth=2, label="Seuil 1.1")
    plt.xlabel("Valeurs de R-hat")
    plt.ylabel("Nombre de paramètres")
    plt.title(title)
    plt.legend()
    plt.show()


