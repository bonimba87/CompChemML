# Graph-level tasks (regression, classification) on molecular databases using Graph Neural Networks


## Project Goal

The aim is to get familiar with molecular graphs and their featurizations, GNN layers, message passing using both node and edge features, and `pytorch-geometric`

The aim is to predict toxic molecular activity using data from the [Tox21 dataset](https://tripod.nih.gov/tox21/challenge/). Molecules are featurized using Morgan (ECFP) fingerprints computed from SMILES strings, and machine learning classifiers are trained to predict toxicity outcomes.

This project is part of an **upskilling roadmap** in cheminformatics and applied machine learning.

##  Dataset

- **Tox21**: A dataset of molecules labeled with 12 different toxicological endpoints (e.g., `SR-MMP`, `NR-AR-LBD`, etc.)
- Labels are multi-label and sparse (not all compounds are tested for all endpoints).

##  Tools & Libraries

- `RDKit` for molecular fingerprints
- `scikit-learn`, `torch`, `pytorch-geometric` for machine learning models
- `matplotlib`  for visualization
- `joblib` for model persistence

##  Methods

- Featurization using 2048-bit Morgan fingerprints (radius 2)
- `Graph Neural Networks`, both hard-coded (without pytorch_geometric) and using the `geometric` package: to add `early stopping` and `batches`
- Model evaluation using ROC AUC
- Optional: multi-label modeling using `MultiOutputClassifier`

##  Results

## Repo Structure


