## Executive Summary

This portfolio is the result of my decision to actively **bridge the gap between my academic background and the skills expected in industry**. While my research experience gave me a strong foundation in computational chemistry and molecular simulations, I recognized that I needed hands-on practice with cheminformatics, machine learning, and physics-based drug discovery tools. So I am building this collection of projects as a way to learn by doing.

The work progresses from **ligand-based QSAR with descriptors and GNNs**, to **structure-based docking**, and finally to **free energy perturbation (FEP)** for relative binding energies. Each project reflects a concrete step in my upskilling journey—designed to demonstrate **initiative, adaptability, and the ability to quickly pick up new methods** that are directly relevant to modern computational chemistry and drug discovery.

**Generative models are in progress**

# Computational Chemistry + Machine Learning Portfolio

## Overview

This portfolio documents my transition from academic computational chemistry to industry-relevant workflows that integrate **machine learning and physics-based modeling**. The projects highlight a progression from descriptor-based QSAR, to graph neural networks (GNNs), and finally to structure-based approaches such as docking and free energy perturbation (FEP).

Projects are grouped into two complementary tracks:

- **Ligand-based modeling**
  - **FoundationalML** — Baseline models with molecular descriptors/fingerprints (toxicity classification, solubility regression).
  - **GNN** — Graph-based deep learning using PyTorch Geometric (solubility prediction and toxicity classification).

- **Structure-based modeling**
  - **SBDD_docking** — Virtual screening workflow with AutoDock Vina (protein prep, ligand docking, scoring, pose selection).
  - **FreeEnergyPerturbation** — Setup of relative binding free energy calculations with OpenFE/OpenMM.

- **Generative modeling**
  - (work in progress)

Through these projects I gained hands-on experience in:
- Molecular featurization and data preprocessing with **RDKit**
- Classical ML and deep learning with **scikit-learn** and **PyTorch/PyG**
- Graph representation learning and message passing for molecules
- Interpreting results in a chemical/biological context
- Setting up and running **FEP workflows** (mapping, alchemical transformations, thermodynamic cycles)
- Performing **structure-based docking** with flexible ligand placement in rigid protein pockets

> This portfolio demonstrates how I combine **data-driven ML** and **physics-based simulations** into end-to-end workflows for molecular property prediction and drug discovery.

---

## Repository Structure
- `FoundationalML/` — Classical descriptor-based QSAR (toxicity, solubility)  
- `GNN/` — Graph neural network models and utilities  
- `SBDD_docking/` — Structure-based docking workflows  
- `FreeEnergyPert_101/` — Free energy perturbation setup and examples  
- `README.md` files in each folder describe project details.

---

## How to Use This Repo

Each notebook is self-contained and runnable independently.

---

## Notes
- The repository is **actively maintained** and will continue to expand (e.g., generative models)
- Any feedback/inquiry is greatly appreciated! Please write to `lorenzobonimba@gmail.com`. Thank you for your help with improving these materials! 
