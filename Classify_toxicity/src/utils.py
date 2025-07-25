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
    
    """ Helper function: compute descriptors from mol object, as from SMILES representation 
        There are human-readable, interpretable and carry physicochemical information

        INPUT: mol (RDKit representation), molecular representation from parsed SMILES string
        OUTPUT: desc, dictionary of physico-chemical properties and their values
    """
 
    desc = {}
    desc['MolWt'] = Descriptors.MolWt(mol)        # here we know the "Descriptors" functions commands explicitly
    desc['LogP'] = Descriptors.MolLogP(mol)
    desc['NumHDonors'] = Descriptors.NumHDonors(mol)
    desc['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    desc['TPSA'] = Descriptors.TPSA(mol)
    desc['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    
    return desc

# Compute ECFP4 fingerprint (bit vector), Morgan fingerptins with a radius of 2
def compute_ecfp4(mol, nBits=2048):
    
    """ Compute 2048 bit fingerprints out of mol
        These are not human readable, not easily interpretable; but capture struturals subgraphs info

        INPUT: mol (RDKit representation), molecular representation from parsed SMILES string
        OUTPUT: arr, binary array of size 2048 with the fingerprint representation
    """
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# featurizer: prepare features for the regression
def featurizer(smiles_list):
    
    """ Take in list of SMILES representation for different molecules, and return Chem descriptors &
        Fingerprints representations
    
        OUTPUT: * features (dict), dictionary of features 
                * valid_feature_idx (list), list of indexes of molecules that are legit (not Nan!)
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
