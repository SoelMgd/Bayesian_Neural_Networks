# Bayesian Neural Networks Analysis with HMC

This project replicates and extends the work of [Wenzel et al. (2020)](https://arxiv.org/abs/2002.03285) on Bayesian Neural Networks (BNN) using Hamiltonian Monte Carlo (HMC) for posterior inference. We aim to assess the performance and convergence properties of HMC applied to deep learning models.

## Key Features:
- Replication of experiments from [Wenzel et al. (2020)](https://arxiv.org/abs/2002.03285) with reduced resources, in pytorch.
- Implementation of HMC for Bayesian inference in neural networks.

## Repo structure:

Notebooks experiments
- 01_prior_posterior_experiments: experiments about prior variance and posterior temperature
- 02_mixing_experiments: experiments about HMC chains mixing: variance ratio and 2D visualization
- 02_mixing_experiments_Hamiltorch: same code but based in Hamiltorch (much faster computation on GPU but need Hamiltorch)
- 03_trajectory_length_experiments: experiments about trajectory lengths on HMC chains

- utils
-- models.py pytorch models used
-- hmc.py HMC sampling algorithm and utils functions
-- eval.py evaluating and vizualizing functions


## Requirements
Basic requirements are needed to make it run, decent versions of : pytorch, numpy, matplotlib, tqdm in your python environment should work

## Caution
Code can work on GPU and is much faster but can lead to CUDA memory errors for to many leapfrog steps.
It works on CPU but is much slower.


