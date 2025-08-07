# Computational Chemistry + Machine Learning Portfolio

## Overview

This portfolio documents my upskilling journey as I work on transitioning from academic research in computational chemistry toward more applied and industry-relevant machine learning workflows. Recognizing the growing importance of ML in drug discovery and chemical property prediction, I explored a variety of cheminformatics tasks — starting from foundational ML models using molecular descriptors and fingerprints, to more complex Graph Neural Networks (GNNs) using molecular graph representations.

Projects are grouped into two tracks:
- **FoundationalML**: Baseline models using molecular descriptors/fingerprints (toxicity classification, solubility regression).
- **GNN**: Graph-based deep learning with `PyG` (solubility prediction and toxicity classification).

Through this work, I gained hands-on experience in:
- Molecular data preprocessing and feature engineering with `RDKit`
- Supervised ML workflows using scikit-learn and `PyTorch`
- Graph representation of molecules and message-passing GNNs
- Visualization and interpretation of results in a chemical context

> This portfolio is a reflection of my effort to bridge the gap between academic research and industry expectations in computational chemistry, cheminformatics, and data-driven modeling.

## Repository Structure

- `FoundationalML/`
  - `Classify_toxicity/`
  - `Predict_logsolubity/`
- `GNN/`
  - GNN-based tasks and utilities
- `README.md` files in each subfolder provide additional details.

## How to Use This Repo

- Start with the foundational models to see descriptor-based workflows.
- Explore the GNN section for deep learning applied to molecular graphs.
- Each notebook is self-contained but shares common utility functions from `src/utils.py`.


