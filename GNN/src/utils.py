import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F   # these are the activation functions

# Set random seed for reproducibility
torch.manual_seed(42)

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

from rdkit.Chem import rdmolops
import networkx as nx
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt


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


def mol_to_nx(mol):

    """ From RDKit molecule object to Network x object: atoms (nodes) & bonds (edges = [i,j] & its attributes) """
    
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                   atomic_num=atom.GetAtomicNum(), 
                   symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        print(bond, bond.GetBondType())
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=str(bond.GetBondType()))
    return G


def visualize_molecular_graph(G, node_size, node_color, smiles):

    """ Plot molecular pure graph, highlight node and edge labels """
    
    # Graph object
    pos = nx.spring_layout(G, seed=42)  # layout automatico
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    
    # Draw node labels
    labels = nx.get_node_attributes(G, 'symbol')    # atom labels
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    # Draw edges, define edge labels and draw them
    nx.draw_networkx_edges(G, pos, width=2)
    edge_labels = {(i, j): str(d["bond_type"]) for i, j, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title(f"Graph of Molecule {smiles}")
    plt.axis('off')


def graph_from_nodes_and_edges(edge_index, atom_types, title="Molecule"):
    G = nx.Graph()

    # Add nodes with atom labels
    for idx, atom in enumerate(atom_types):
        G.add_node(idx, label=atom)

    # Add edges
    for src, tgt in edge_index:
        G.add_edge(src, tgt)

    # Get labels
    labels = nx.get_node_attributes(G, 'label')
    
    # Layout
    pos = nx.spring_layout(G, seed=42)  # deterministic layout
    
    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=1200, font_size=14)
    nx.draw_networkx_edges(G, pos, width=2)
    plt.title(title)
    plt.axis('off')
    plt.show()

def atom_features(atom):

    """
    Extract atom-specific (NODE) features, have them arrayed in an array and convert it to pytorch tensor

    INPUT: * atom, element of mol.GetAtoms() list in RDKit
    OUTPUT: * pytorch tensor with features for that atom
    """
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())], dtype=torch.float)

def bond_features(bond):

    """
    Extract bond-specific (EDGE) features, have them arrayed in an array and conver to pytorch tensor

    INPUT: * bond, element of mol.GetBonds() list in RDKit
    OUTPUT: * pytorch tensor with features for that bond
    """
    bt = bond.GetBondType()
    return torch.tensor([int(bt == Chem.rdchem.BondType.SINGLE),
                    int(bt == Chem.rdchem.BondType.DOUBLE),
                    int(bt == Chem.rdchem.BondType.TRIPLE),
                    int(bt == Chem.rdchem.BondType.AROMATIC)], dtype=torch.float)


class GNNLayer(nn.Module):   #base class for all neural network components in pytorch
    
    def __init__(self, in_dim, out_dim):   # constructor
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)      # The Linear layer is our learnable function that updates node embeddings after aggregating messages.
                                                      # e' la classica y = xW.T + b delle reti fully connected
    def forward(self, x, edge_index):   # this function MUST be overridded everytime
        """
        x: [num_nodes, in_dim]
        edge_index: [num_edges, 2], where each row is [source, target]

        This is the key method that does the **message passing** and **embedding update**.
        This details how data flows from one layer to the next
        """
        
        num_nodes = x.shape[0]
        messages = torch.zeros(num_nodes, x.shape[1])

        # message passing + basic aggregating (= sum!); just node embeddings
        for src, tgt in edge_index:
            messages[tgt] += x[src]  # Add message from src to tgt: messages[i] will hold the sum of all messages (here, node embeddings) received by node i from its neighbors.

        """This is the UPDATE step:

        Each node has received messages from neighbors (i.e., messages[i]). 
        We transform those aggregated messages using a linear layer (i.e., a fully connected layer with weights and bias).
        This is a learnable transformation to produce updated node embeddings"""
        
        out = self.linear(messages)
        out = F.relu(out)    # increase expressivity, questo e' quello che c'e' nelle reti normali RElu(linear transfomration) to
                             # introduce non-linearity in the representation
        return out

""" Output is the updated node embeddings after this GNN layer. These can be fed to: another GNN layer, a pooling layer
(to get a graph embedding), a classifier or regressor"""

# === Define Graph-level GNN model ===

class GraphClassifier(nn.Module):

    """
    This defines a PyTorch model that:
    - Takes a molecular graph as input (nodes + edges)
    - Uses 2 GNN layers to update atom (node) embeddings
    - Pools those embeddings into a single graph embedding
    - Passes that through an linear
    - Returns a scalar (to be later smoothed wiht a sigmoid to map to either 0 or 1)

    Please note that the graph connectivity is never changed here, just node (possibly even edge) embeddings are propagated
    """
    
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        # here I instantiate the layers
        self.gnn1 = GNNLayer(in_dim, hidden_dim)  # message passing + Relu(): node embedding from (in_dim) to (hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)   # message passing + Relu(): node embedding from (hidden_dim) to (hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)   # takes graph level embedding and maps it to a logit only
                                                     # which will then be sqashed to 1 or 0 by a sigmoid (classification)

    def forward(self, node_feats, edge_index):   # define the architecture

        # here I call the layers
        h = self.gnn1(node_feats, edge_index)    # it automatically call the .forward() method behind the scene
        h = self.gnn2(h, edge_index)

        # Sum pooling: [num_nodes, hidden_dim] → [1, hidden_dim], from node to graph embedding, this represents the whole molecule
        graph_embedding = h.sum(dim=0, keepdim=True)          # final pooling: collect all nodes into a graph thing 
                                                            # a node classifier would not have this
        """In GNNs, after several message-passing layers, each node has its own learned embedding vector 
                                                            — a dense feature representation summarizing its local neighborhood. But often, you want a single 
                                                            vector representing the entire graph (e.g., a molecule) for tasks like classification or regression."""
        out = self.classifier(graph_embedding)  # output is [1, 1], a logit
        
        return out.squeeze(1)    # remove one of the 1 dimensions, to make calculation easier down the line
