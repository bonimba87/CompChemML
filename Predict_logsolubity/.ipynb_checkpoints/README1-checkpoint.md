# 🧪 Molecular Solubility Prediction using Descriptors and Fingerprints

This project predicts aqueous solubility (logS) of small molecules using cheminformatics features derived from SMILES strings. The dataset used is **ESOL**, a curated solubility dataset.

## 🎯 Goal

Compare performance of molecular descriptors and ECFP4 fingerprints in predicting molecular solubility via machine learning regression models (Random Forest, XGBoost).

## 📦 Dataset

- **Name:** ESOL (Delaney dataset)
- **Source:** MoleculeNet via DeepChem or [CSV download](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)
- **Size:** ~1128 molecules
- **Target:** log solubility in mol/L

## 🧰 Tools & Libraries

- `RDKit` – molecular featurization (descriptors + ECFP)
- `scikit-learn`, `XGBoost` – ML models
- `matplotlib` – plotting
- `pandas`, `numpy` – data handling

## 🧪 Features used

- **Descriptors:** MolWt, LogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds
- **Fingerprints:** ECFP4 (Morgan fingerprints with radius 2, 2048 bits)

## 🧠 Models & Evaluation

- Trained **Random Forest** and **XGBoost** regressors
- Metrics:
  - cross validation
  - RMSE
  - R² score
- Feature importance visualized for interpretability

## 📈 Results

- Compare optimized  **Random Forest** and **XGBoost** regressor performances 
| Model       | RMSE    | R²   |
|-------------|---------|------|
| RF          |  0.80   | 0.87 |
| XGBoos      |  0.75   | 0.88 |

- Explore optimized  **Random Forest** regressor performances
| Model       | Input           | RMSE  | R²   |
|-------------|------------------|--------|------|
| RF          | Descriptors       | 0.83   | 0.85 |
| RF          | Fingerprints      | 1.16   | 0.71 |
| RF          | Descriptors + FP  | 0.68   | 0.81 |


## 📊 Visualizations

- Predicted vs Actual logS
- Top 10 most important features (Random Forest/XGBoost)
![Predicted vs Actual Solubility](images/pred_vs_true.png)

## 📚 What I Learned

- Difference between physicochemical descriptors and structural fingerprints
- Handling molecular data (SMILES → features)
- Training and evaluating regression models in cheminformatics
- Feature interpretation for model explainability

## 🧭 Next Steps

- Try neural network regression (e.g., MLP)
- Compare early/late fusion of descriptors + fingerprints


## 📎 References

- Delaney, J.S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *J. Chem. Inf. Comput. Sci.* **44**, 1000–1005 (2004).
- [MoleculeNet paper](https://arxiv.org/abs/1703.00564)
- [DeepChem docs](https://deepchem.io/)
