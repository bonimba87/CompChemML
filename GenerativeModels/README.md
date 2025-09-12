# Generative Models for Molecules

This folder collects exploratory projects on **generative modeling approaches applied to molecular data**.  
The goal is to gain hands-on familiarity with modern techniques that are increasingly relevant in computational drug discovery, and to complement the ligand-based (QSAR, GNNs) and structure-based (docking, FEP) sections of this portfolio.

---

## Motivation
Generative models are reshaping early-stage drug design by enabling the **proposal of novel chemical structures** rather than only predicting properties of existing molecules.  
These methods connect well with my background in ML and graph-based learning, while allowing me to explore the frontier of molecular design.

In this section, I aim to:
- Understand the **foundational techniques** (VAE, normalizing flows, diffusion models).  
- Implement **simplified, educational versions** of these methods using small datasets (QM9, subsets of ChEMBL).  
- Learn and apply **best practices** for evaluating generated molecules:  
  - **Validity** (chemically correct molecules)  
  - **Uniqueness** (non-duplicates)  
  - **Novelty** (not in the training set)  
  - **Drug-likeness metrics** (QED, logP)

---

## Planned Techniques & Notebooks (so far)
1. **Variational Autoencoders (VAE)**  
   - Learn latent continuous representations of SMILES.  
   - Generate new molecules by sampling from latent space.  
   - Notebook: `01_VAE_SMILES.ipynb`



---

## Industry Relevance
These approaches are central to **modern AI-driven drug discovery pipelines**:
- **VAE**: foundation for many molecular design platforms.  
- **Normalizing Flows**: provide explicit likelihoods, useful for property-driven design.  
- **Diffusion Models**: currently *state of the art* in generative modeling for chemistry and materials.  
- **Conditional Generation**: demonstrates how generative models can be aligned with medicinal chemistry objectives.  

---

## Status
 This section is under active development.  
Each notebook will include:
- A brief theoretical overview.  
- Implementation and training loop.  
- Evaluation of generated molecules with RDKit.  
- Reflections: *"What I learned"*.  


