# Toxicity Classification using Molecular Fingerprints

This project explores binary and multi-label classification of toxic compounds using cheminformatics datasets and machine learning.

## Project Goal

The aim is to predict toxic molecular activity using data from the [Tox21 dataset](https://tripod.nih.gov/tox21/challenge/). Molecules are featurized using Morgan (ECFP) fingerprints computed from SMILES strings, and machine learning classifiers are trained to predict toxicity outcomes.

This project is part of an **upskilling roadmap** in cheminformatics and applied machine learning.

##  Dataset

- **Tox21**: A dataset of molecules labeled with 12 different toxicological endpoints (e.g., `SR-MMP`, `NR-AR-LBD`, etc.)
- Labels are multi-label and sparse (not all compounds are tested for all endpoints).

##  Tools & Libraries

- `RDKit` for molecular fingerprints
- `scikit-learn` for machine learning models
- `matplotlib`  for visualization
- `joblib` for model persistence

##  Methods

- Featurization using 2048-bit Morgan fingerprints (radius 2)
- Classification using `RandomForestClassifier`
- Model evaluation using ROC AUC
- Optional: multi-label modeling using `MultiOutputClassifier`

##  Results

- Basic binary classification on selected tasks (e.g., `SR-MMP`)
- Model performance is modest (ROC AUC ~0.6–0.7), limited by class imbalance and lack of tuning
- Feature importances extracted and visualized
- Repository includes saved models and performance plots

## Repo Structure


