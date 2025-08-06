import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


def compute_descriptors(mol):
    
    """
    Compute a set of interpretable physicochemical descriptors for a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object, typically parsed from a SMILES string.

    Returns
    -------
    desc : dict
        Dictionary of computed descriptors, including:
        - Molecular weight (MolWt)
        - Octanol-water partition coefficient (LogP)
        - Number of hydrogen bond donors (NumHDonors)
        - Number of hydrogen bond acceptors (NumHAcceptors)
        - Topological polar surface area (TPSA)
        - Number of rotatable bonds (NumRotatableBonds)

    Notes
    -----
    These descriptors are widely used in cheminformatics to encode 
    physicochemical properties relevant to solubility, permeability, and bioavailability.
    """
 
    desc = {}
    desc['MolWt'] = Descriptors.MolWt(mol)        # here we know the "Descriptors" functions commands explicitly
    desc['LogP'] = Descriptors.MolLogP(mol)
    desc['NumHDonors'] = Descriptors.NumHDonors(mol)
    desc['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    desc['TPSA'] = Descriptors.TPSA(mol)
    desc['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    
    return desc

def compute_ecfp4(mol, nBits=2048):
    
    """
    Compute the Extended-Connectivity Fingerprint (ECFP4) for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object, typically parsed from a SMILES string.
    nBits : int, optional (default=2048)
        Length of the generated fingerprint vector.

    Returns
    -------
    arr : np.ndarray
        A 1D binary NumPy array of length `nBits` representing the ECFP4 fingerprint.

    Notes
    -----
    ECFP4 (Morgan fingerprints with radius=2) capture local substructures around atoms.
    These features are not human-interpretable but are highly predictive in machine learning models.
    """
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# featurizer: prepare features for the regression
def featurizer(smiles_list):

    """
    Generate molecular descriptors and ECFP4 fingerprints from a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES representations of molecules.

    Returns
    -------
    features : list of dict
        Each element is a dictionary containing:
        - Physicochemical descriptors (e.g., MolWt, TPSA, LogP)
        - Binary ECFP4 fingerprint features with keys like 'ECFP_0', 'ECFP_1', ...

    valid_feature_idx : list of int
        Indices of SMILES strings that were successfully parsed into RDKit Mol objects.

    Notes
    -----
    - Invalid SMILES are skipped with a warning printed to console.
    - This function is useful when training ML models on tabular molecular features.
    """
    
    features = []
    valid_feature_idx = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = compute_descriptors(mol)   # this is a dictionary already
            fp = compute_ecfp4(mol)
            desc.update({f'ECFP_{j}': fp[j] for j in range(len(fp))})   # add fingerprint dictionary entries
            features.append(desc)
            valid_feature_idx.append(i)
        else:
            print(f"Invalid SMILES skipped: {smi}")
    return features, valid_feature_idx
