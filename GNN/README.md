# Graph-level tasks (regression, classification) on molecular databases using Graph Neural Networks

## Project Goal

The aim is to get hands-on experience with graph representation of molecules, their featurizations, GNN layers, message passing using both node and edge features, and `PyG`.

The aim is to predict toxic molecular activity using data from the [Tox21 dataset](https://tripod.nih.gov/tox21/challenge/) and predict log Solubility of molecules using [ESOL] dataset (tasks we already addressed using foundational ML tools) using Graph Neural Networks, leveragin the inherent graph-like organization of the input.

Molecules are featurized/represented as molecular graphs by listing the node indices (and their embeddings) and edge indices (and their embeddings/attributes). GNN exploit the connectivity of the molecules explicitly, and we expect them to be more efficient at tacking ChemInfo/Molecular modeling problems. 

This project is part of an **upskilling roadmap** in cheminformatics and applied machine learning.

## Note on environment
Notebooks have been run on `Google Colab`, which makes installing `PyG` much easier than on a Mac. Path to `utils.py` and input_data files have to be edited accordingly.
 
##  Dataset

- **Tox21**: A dataset of molecules labeled with 12 different toxicological endpoints (e.g., `SR-MMP`, `NR-AR-LBD`, etc.)
- **ESOL**: A dataset of molecules with their log aqueous solubility

##  Tools & Libraries

- `RDKit` for converting molecules into molecular graph
- `scikit-learn`, `torch`, `PyG` for model definition, instantiation, training and testing
- `matplotlib`  for visualization
- `joblib` for model persistence
- `src/utils.py`: list of auxiliary functions and classes that are imported in the main notebooks

##  Methods

- Featurization using `RDKit`-based atom and bond features
- `Graph Neural Networks`:
     * Hard-coded using `torch`; simple classification task ("Is there an Oxygen atom in the molecule?") on a user-defined hard coded datset, allowing for node and/or edge message passing ['Intro_molecules_as_graphs.ipynb]
     * Used standard `PyG` package (task: "Is the molecule toxic wrt a given endpoint?"), use `early stopping` and `batches`; both `GraphConv` and `NNConv` architectures to control edge-modulated message passing (off v on) [GNN_ToxicityClassifier.ipynb]
     * Regression problem ("what is the molecule acqueous solubility?"), just by changing the architecture head from a classifier to a regressor [GNN_SolubilityRegressor_colab.ipynb]
- Model evaluation using R^2, RMSE (regressor) and ROC AUC (classifier)

____

## Learning highlights

- Implemented graph featurization from SMILES using RDKit
- Learned foundational principles of GNNs (graph representation, message passing, graph-level/node-level tasks)
- Coded a GNN from scratch in `torch`; including edge-passin logic

- Learned `pygeometric` syntax and practice when it comes to preparing the data for training and testing (`Data`, `DataLoader`, `Batch`)
- Used `pygeom`-based `GraphConv` and `NNConv` and `global_mean` to define legit GNN architectures for classification and regression; allowed both for edge modulated passing and node only passing
- Modularized code for readability and reuse (`utils.py`)

## Expansions

- [] Understand how **node-dependent prediction tasks** are performed/implemented in `torch-geometric` (e.g., atom classification, aromaticity)



