# Bayesian Neural Networks Analysis with HMC

This project replicates and extends the work of [Wenzel et al. (2020)](https://arxiv.org/abs/2002.03285) on Bayesian Neural Networks (BNN) using Hamiltonian Monte Carlo (HMC) for posterior inference. We aim to assess the performance and convergence properties of HMC applied to deep learning models.

## Repository Structure

### Notebooks (Experiments)
- **`01_prior_posterior_experiments`**: Experiments on prior variance and posterior temperature.
- **`02_mixing_experiments`**: Experiments on HMC chain mixing, including variance ratio and 2D visualization.
- **`02_mixing_experiments_Hamiltorch`**: Same experiments as above, but implemented using **Hamiltorch** for faster GPU computations.
- **`03_trajectory_length_experiments`**: Experiments on trajectory lengths in HMC chains.

### Utils
- **`models.py`**: PyTorch models used in the experiments.
- **`hmc.py`**: HMC sampling algorithm and utility functions.
- **`eval.py`**: Evaluation and visualization functions.


## Requirements
Basic requirements are needed to make it run, decent versions of : pytorch, numpy, matplotlib, tqdm in your python environment should work
You can use our version with pip install -r requirements.txt


## Caution
Code can work on GPU and is much faster but can lead to CUDA memory errors for to many leapfrog steps.
It works on CPU but is much slower.
