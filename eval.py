import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from hmc import bma_inference, set_model_params, flatten_params, unflatten_params


def evaluate_models(model, test_loader, samples, device="cuda", n_classes=10, prior_std=1, verbose=False):
    """
    Évalue les performances d'unce chaine échantillonée sur un ensemble de test en utilisant BMA et calcule les courbes de calibration.

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
    to_return['ECE'] = results['ECE']
    to_return['log_likelihood'] = -F.cross_entropy(outputs, targets, reduction='sum').item()
    to_return['log_prior'] = -0.5 * sum(torch.sum(p ** 2) for p in model.parameters()) / (prior_std ** 2)
    to_return['log_posterior'] = to_return['log_likelihood'] + to_return['log_prior']

    if verbose:
        # Affichage des résultats
        print("\nAccuracy : ", to_return['accuracy'] )
        print("Confidence : ", to_return['confidence'])
        print("ECE : ", to_return['ECE'])
        print("log prior : ", to_return['log_prior'])
        print("log likelihood : ", to_return['log_likelihood'])
        print("log posterior : ", to_return['log_posterior'])

    return to_return

def evaluate_single_model(model, test_loader, sample, device="cuda", n_classes=10, prior_std=1., verbose=False):
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
        'ECE': results['ECE'],
        'log_likelihood': -F.cross_entropy(outputs, targets, reduction='sum').item()
    }

    to_return['log_likelihood'] = -F.cross_entropy(outputs, targets, reduction='sum').item()
    to_return['log_prior'] = -0.5 * sum(torch.sum(p ** 2) for p in model.parameters()) / (prior_std ** 2)
    to_return['log_posterior'] = to_return['log_likelihood'] + to_return['log_prior']

    if verbose:
        print("\nAccuracy : ", to_return['accuracy'])
        print("Confidence : ", to_return['confidence'])
        print("ECE : ", to_return['ECE'])
        print("log likelihood : ", to_return['log_likelihood'])
        print("log prior : ", to_return['log_prior'])
        print("log likelihood : ", to_return['log_likelihood'])
        print("log posterior : ", to_return['log_posterior'])

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
        "ECE": ece
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


def plot_grid(test_loader, model, w1, w2, w3, grid_size, n_classes=10, max_nll=1000000, device='cpu', verbose=False):
    w1_flat = flatten_params(w1)
    w2_flat = flatten_params(w2)
    w3_flat = flatten_params(w3)
    
    a_range = torch.linspace(-1, 2, grid_size)
    b_range = torch.linspace(-1, 2, grid_size)

    # Construire la grille de paramètres
    theta_grid = torch.zeros(grid_size, grid_size, w1_flat.shape[0], device=device)
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            theta_grid[i, j] = a * w1_flat + b * w2_flat + (1 - a - b) * w3_flat

    # Initialisation des grilles
    nll_grid = torch.zeros(grid_size, grid_size, device=device)
    log_prior_grid = torch.zeros(grid_size, grid_size, device=device)
    log_posterior_grid = torch.zeros(grid_size, grid_size, device=device)

    for i in tqdm(range(grid_size * grid_size), desc="Evaluating samples"):
        sample = unflatten_params(theta_grid[i // grid_size, i % grid_size], model)
        results = evaluate_single_model(model, test_loader, sample, device=device, n_classes=n_classes, verbose=verbose)

        nll = - results['log_likelihood']
        log_prior = results['log_prior']
        log_posterior = results['log_posterior']

        log_prior_grid[i // grid_size, i % grid_size] = log_prior
        log_posterior_grid[i // grid_size, i % grid_size] = log_posterior        
        if torch.isnan(torch.tensor(nll)) or nll > max_nll:
            nll_grid[i // grid_size, i % grid_size] = max_nll
        else:
            nll_grid[i // grid_size, i % grid_size] = nll

    # Conversion en numpy
    nll_grid_np = nll_grid.cpu().detach().numpy()
    log_prior_np = log_prior_grid.cpu().detach().numpy()
    log_posterior_np = log_posterior_grid.cpu().detach().numpy()

    # Affichage des trois heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # NLL
    im1 = axes[0].contourf(a_range.cpu().numpy(), b_range.cpu().numpy(), nll_grid_np, levels=50, cmap="viridis")
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_title("Negative Log-Likelihood (NLL)")
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("b")

    # Log-prior
    im2 = axes[1].contourf(a_range.cpu().numpy(), b_range.cpu().numpy(), log_prior_np, levels=50, cmap="plasma")
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_title("Log-Prior")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("b")

    # Log-posterior
    im3 = axes[2].contourf(a_range.cpu().numpy(), b_range.cpu().numpy(), log_posterior_np, levels=50, cmap="cividis")
    fig.colorbar(im3, ax=axes[2])
    axes[2].set_title("Log-Posterior")
    axes[2].set_xlabel("a")
    axes[2].set_ylabel("b")

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

    # Estimation ajustée de la variance totale de chaque paramètre
    sigma_plus = ((N - 1) / N) * W + (B / N)

    # Calcul final de R-hat
    R_hat = ((M + 1) / M) * (sigma_plus / W) - (N - 1) / (M * N)  # (D,)

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
    #plt.axvline(1.2, color='r', linestyle='dashed', linewidth=2, label="1.2 treshold")
    plt.xlabel("R values")
    #plt.ylabel("Nombre de paramètres")
    plt.title(title)
    plt.legend()
    plt.show()

def prepare_chains(model, test_loader, chains, device='cpu'):
    chains_output = []
    for i, chain in enumerate(chains):
        chain_predictions = []
        
        for sample in tqdm(chain, desc="Evaluating chain"):
            set_model_params(model, sample, device)
            model.to(device)
            model.eval()

            preds=[]
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc="Predicting the test set"):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    preds.append(F.softmax(output, dim=1)) 
            preds_tensor = torch.cat(preds, dim=0) # (num_samples, num_classes)
            chain_predictions.append(preds_tensor) # (N, num_samples, num_classes)
        
        chains_output.append(torch.stack(chain_predictions))

    chains_output = torch.stack(chains_output)
    return chains_output

def compute_r_hat_per_image(chains):
    """
    Calcule la statistique R-hat de Gelman & Rubin pour chaque image du test set dans l'espace des prédictions.

    Paramètres :
        chains (torch.Tensor) : tenseur de forme (M, N, num_samples, num_classes)
                                contenant les prédictions softmax pour chaque chaîne (M), chaque échantillon (N),
                                chaque image du test set (num_samples) et chaque classe (num_classes).

    Retourne :
        torch.Tensor : valeurs de R-hat pour chaque image du test set.
    """
    M, N, num_samples, num_classes = chains.shape  # (M=3, N=num_samples, num_images, num_classes)
    psi_moyenne = chains.mean(dim=1)  # Moyenne intra-chaîne des prédictions pour chaque image et pour chaque classe
    psi_global = psi_moyenne.mean(dim=0)  # Moyenne globale des prédictions pour chaque image et pour chaque classe
    B = (N / (M - 1)) * ((psi_moyenne - psi_global) ** 2).sum(dim=0)  # Variance inter-chaîne B pour chaque image et pour chaque classe
    W = (1 / (M * (N - 1))) * ((chains - psi_moyenne[:, None, :, :]) ** 2).sum(dim=(0, 1, 2))  # Variance intra-chaîne W pour chaque image et pour chaque classe
    sigma_plus = ((N - 1) / N) * W + (B / N)  # (num_samples, num_classes) # Estimation ajustée de la variance totale pour chaque image et chaque classe

    # Calcul de R-hat pour chaque image
    R_hat = ((M + 1) / M) * (sigma_plus / W) - (N - 1) / (M * N)  # (num_samples, num_classes)

    # R-hat moyen sur toutes les classes pour chaque image
    R_hat_per_image = R_hat.mean(dim=-1)  # (num_samples)

    return R_hat_per_image