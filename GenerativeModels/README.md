# Generative Models for Molecules

This folder collects exploratory projects on **generative modeling approaches applied to molecular data**.  
The goal is to gain hands-on familiarity with modern techniques that are increasingly relevant in computational drug discovery, and to complement the ligand-based (QSAR, GNNs) and structure-based (docking, FEP) sections of this portfolio. 

These projects explore how modern generative models learn to produce new data by sampling from a learned probability distribution. Whether through the latent variable formulation of VAEs, the adversarial setup of GANs, or the denoising process of Diffusion Models, the central idea is the same: start from random noise and gradually recover structure, by applying a transformation decoded by a neural network.

---

## Motivation
Generative models are reshaping early-stage drug design by enabling the **proposal of novel chemical structures** rather than only predicting properties of existing molecules.  
These methods connect well with my background in probability thery, statistical mechanicsm ML and graph-based learning, while allowing me to explore/probe the frontier of molecular design.

In this section, I aim to:
- Understand the **foundational techniques** (VAE,  diffusion models).  
- Implement **simplified, educational versions** of these methods using small datasets (QM9, subsets of ChEMBL).  
- Learn and apply **best practices** for evaluating generated molecules:  
  - **Validity** (chemically correct molecules)  
  - **Uniqueness** (non-duplicates)  
  - **Novelty** (not in the training set)  
  - **Drug-likeness metrics** (QED, logP)

---

## Planned Techniques & Notebooks
0. **Recurrenct Neural Networks (RNN)**
   - Use RNNs to sample new SMILES representations using autoregressive sampling
   - Validated models using validity, uniqueness and novelty
   - Notebook: `RNN_SMILES.ipynb`

1. **Variational Autoencoders (VAE)**  
   - Learn latent continuous representations of SMILES.  
   - Generate new molecules by sampling from latent space.  
   - Notebook: `VAE_SMILES.ipynb`

2. **Diffusion Models (DDPM)**  
   - Understand iterative denoising processes for  generation.  
   - Implement a simplified diffusion 2D model (like, MNIST) (ongoing)  
   - Implement a SMILES diffusion (planned)

---

## Industry Relevance
These approaches are central to **modern AI-driven drug discovery pipelines**:
- **VAE**: foundation for many molecular design platforms.  
- **Diffusion Models**: currently state of the art in generative modeling for chemistry and materials.  

---

## Outlook
Probe normalizing flows, conditional generators  

