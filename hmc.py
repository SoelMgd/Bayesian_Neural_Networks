import torch
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
import numpy as np


# Fonction pour initialiser le modèle suivant le prior
def initialize_weights(model, mean=0.0, std=1):
    for param in model.parameters():
        if param.requires_grad:  # S'assurer que les paramètres sont appris par le modèle
            init.normal_(param, mean=mean, std=std)

# Fonction pour mettre à jour les paramètres du modèle
def set_model_params(model, theta):
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), theta):
            param.copy_(new_param)

# Fonction de log-probabilité avec prior gaussien
def log_prob_func(model, data, target, prior_std=1.0, temperature=1.):
    logits = model(data)
    log_likelihood = -F.cross_entropy(logits, target, reduction='sum')
    prior = -0.5 * sum(torch.sum(p ** 2) for p in model.parameters()) / (prior_std ** 2)
    return log_likelihood + prior / temperature

# Fonction pour collecter les gradients
def compute_gradients(model, data, target, temperature=1.):
    log_prob = log_prob_func(model, data, target, temperature=temperature)
    grads = torch.autograd.grad(log_prob, model.parameters()) #, create_graph=True)
    return grads

# Implémentation de Leapfrog
def leapfrog(theta, r, step_size, num_steps, model, data, target, temperature=1.):
    theta = [p.clone().detach().requires_grad_(True) for p in theta]
    r = [ri.clone().detach() for ri in r]

    set_model_params(model, theta)
    grad = compute_gradients(model, data, target, temperature=temperature)

    for i in range(len(r)):
        r[i] = r[i] + 0.5 * step_size * grad[i]

    for _ in range(num_steps):
        for i in range(len(theta)):
            theta[i] = theta[i] + step_size * r[i]

        set_model_params(model, theta)
        grad = compute_gradients(model, data, target, temperature=temperature)

        for i in range(len(r)):
            r[i] = r[i] + step_size * grad[i]

    set_model_params(model, theta)
    grad = compute_gradients(model, data, target, temperature=temperature)

    for i in range(len(r)):
        r[i] = r[i] - 0.5 * step_size * grad[i]

    return theta, r

# Fonction d'acceptation Metropolis-Hastings
def acceptance(theta, r, new_theta, new_r, model, data, target, device):
    set_model_params(model, theta)
    current_H = -log_prob_func(model, data, target) + 0.5 * sum(torch.sum(ri**2) for ri in r)

    set_model_params(model, new_theta)
    proposed_H = -log_prob_func(model, data, target) + 0.5 * sum(torch.sum(ri**2) for ri in new_r)

    accept_prob = torch.exp(current_H - proposed_H)

    if torch.rand(1).to(device) < accept_prob:
        #print(accept_prob)
        return new_theta, 1
    else:
        return theta, 0

def HMC_sampling(device, model, theta, train_loader, step_size, num_steps, n_burnin, n_samples, temperature=1.):
    samples, num_acceptations = [], 0
    
    # Burnin phase
    for _ in tqdm(range(n_burnin), desc="Burn in phase"):
        r = [torch.randn_like(p).to(device) for p in theta]  # momentum
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)  
        theta, r = leapfrog(theta, r, step_size, num_steps, model, data, target, temperature=temperature) # leapfrog

    # Sampling phase
    for _ in tqdm(range(n_samples), desc="Sampling phase"):
        r = [torch.randn_like(p).to(device) for p in theta]  # momentum
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device) 
    
        new_theta, new_r = leapfrog(theta, r, step_size, num_steps, model, data, target, temperature=temperature) # leapfrog
        
        theta, acceptation = acceptance(theta, r, new_theta, new_r, model, data, target, device)
        num_acceptations += acceptation

        del r, new_theta, new_r
        torch.cuda.empty_cache()
        
        samples.append([p.clone().detach().cpu() for p in theta])  # Save samples
    
    print(f"Sampling finished. {len(samples)} samples with {num_acceptations/n_samples} acceptance_ratio.")
    return samples

# Fonction d'inférence avec la méthode BMA
def bma_inference(model, samples, data, device):
    preds = []
    for sample in samples:
        # Charger les paramètres dans le modèle
        for param, s in zip(model.parameters(), sample):
            param.data = s.data  # Mettre à jour les paramètres du modèle avec l'échantillon

        model.to(device)
        output = model(data)
        preds.append(F.softmax(output, dim=1))  # Utilisation de softmax pour obtenir les probabilités

    # Calcul de la moyenne des prédictions
    avg_preds = torch.stack(preds).mean(0)  # Moyenne sur tous les échantillons (axis 0)

    # Retourner la classe avec la plus haute probabilité
    return avg_preds, avg_preds.argmax(dim=1)


def flatten_chain(samples_chain):
    """
    Transforme une liste de listes de tenseurs en un tenseur (N, D).
    
    Paramètres :
        samples_chain (list[list[torch.Tensor]]): Liste de N échantillons, 
            où chaque échantillon est une liste de tenseurs (les poids du modèle).
    
    Retourne :
        torch.Tensor : Tenseur de forme (N, D), où D est la taille totale des poids.
    """
    flattened_samples = []
    
    for sample in samples_chain:  # Parcourir N échantillons
        flat_sample = torch.cat([p.view(-1) for p in sample])  # Aplatir et concaténer
        flattened_samples.append(flat_sample)
    
    return torch.stack(flattened_samples)  # Taille (N, D)

def sample_multiple_chains(model, device, train_loader, num_chains=3, n_samples=200, n_burnin=20, step_size=0.0005, n_leapfrog=20, prior_var=1/5):
    """
    Effectue un échantillonnage HMC pour un nombre donné de chaînes et retourne un tenseur (M, N, D).

    Paramètres :
        device (torch.device) : appareil sur lequel exécuter le modèle (GPU ou CPU)
        train_loader (DataLoader) : chargeur de données pour l'entraînement
        num_chains (int) : nombre de chaînes à exécuter
        n_samples (int) : nombre d'échantillons par chaîne
        n_burnin (int) : nombre d'itérations de burn-in
        step_size (float) : taille des pas de leapfrog
        n_leapfrog (int) : nombre d'étapes leapfrog

    Retourne :
        torch.Tensor : tenseur des échantillons de taille (M, N, D)
    """
    
    prior_std = np.sqrt(prior_var)
    
    all_chains = []

    for i in range(num_chains):
        print(f"🔄 Exécution de la chaîne {i+1}/{num_chains}...")

        # Initialisation d'un modèle ResNet avec les poids
        initialize_weights(model, mean=0.0, std=prior_std)

        # Extraire les poids initiaux
        theta = [p.clone().detach().to(device) for p in model.parameters()]

        # Exécuter HMC pour obtenir des échantillons
        samples = HMC_sampling(device, model, theta, train_loader, step_size, n_leapfrog, n_burnin, n_samples)

        # Transformer en un tenseur (N, D) et envoyer sur CPU
        chain_tensor = flatten_chain(samples).cpu()

        all_chains.append(chain_tensor)

    # Empiler les chaînes en un tenseur (M, N, D)
    samples_weight = torch.stack(all_chains)  # (M, N, D)

    print(f"✅ Échantillonnage terminé. Tenseur final : {samples_weight.shape}")

    return samples_weight


def get_function_space_samples(model, test_loader, samples_weight, device="cuda"):
    """
    Génère les prédictions softmax pour chaque échantillon de poids HMC.

    Paramètres :
        model (torch.nn.Module) : Le modèle PyTorch.
        test_loader (DataLoader) : DataLoader pour le test set.
        samples_weight (torch.Tensor) : Poids HMC de forme (M, N, D).
        device (str) : "cuda" ou "cpu".

    Retourne :
        torch.Tensor : échantillons de prédictions (M, N, C).
    """
    M, N, D = samples_weight.shape
    model.to(device)
    
    # Initialiser un tableau pour stocker les prédictions softmax
    samples_function = []

    for m in range(M):
        chain_predictions = []
        for n in range(N):
            # Charger les poids dans le modèle à partir de samples_weight en utilisant set_model_params
            set_model_params(model, samples_weight[m, n].cpu())  # Déplacer sur CPU pour éviter d'utiliser la mémoire GPU

            # Propager un batch de test à travers le modèle
            model.eval()  # Mettre le modèle en mode évaluation pour désactiver les effets de dropout, batchnorm, etc.
            with torch.no_grad():  # Pas de calcul de gradients
                for data, _ in test_loader:
                    data = data.to(device)
                    logits = model(data)
                    softmax_preds = torch.nn.functional.softmax(logits, dim=-1)
                    chain_predictions.append(softmax_preds.cpu())  # Déplacer sur CPU

        # Stocker les résultats pour la chaîne actuelle
        samples_function.append(torch.stack(chain_predictions))

    return torch.stack(samples_function)  # Forme (M, N, C)

