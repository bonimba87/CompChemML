import numpy as np

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F   # these are the activation functions
from torch.utils.data import random_split

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

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv


def split_data(full_data, train_ratio, val_ratio):

    """
    Helper function, split dataset into train, validation and test set, and define batches

    INPUT: * full_data, torch Data object, featurization (and output) for all molecular graphs
           * train_ratio (float): percentage of all data to allocate for training
           * val_ratio (float): percentage of all data to allocate for validation

    OUTPUT: * train_dataset (pytorch  Data object): training dataset
            * val_dataset (pytorch  Data object): validation dataset
            * test_dataset (pytorch Data object): test dataset
    """

    total_size = len(full_data)
    train_size = int(train_ratio * total_size)
    val_size   = int(val_ratio * total_size)
    test_size  = total_size - train_size - val_size  # handles rounding

    train_dataset, test_dataset, val_dataset = random_split(full_data, [train_size, test_size, val_size])

    return train_dataset, val_dataset, test_dataset

    

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

def graph_featurizer_pygeom(mol, mol_id, y, edge_attrib = None):

    """A molecule is translated into a featurized graphs, with nodes and node labels + edges and edge labels (if applicable) """

    atom_feats = []
    edge_index = []
    edge_attr = []

    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))     # list of torch tensors

    for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # graph is undirected

            if edge_attrib is not None:
                edge_attr.append(bond_features(bond))
                edge_attr.append(bond_features(bond)) # if bidirectional, we miss half the bonds if we don't include the flipped one
                                                  # clearly here [i,j] and [j,i] share the same feature
    # from list of torch tensors to one torch tensor
    x = torch.stack(atom_feats)

    if edge_attrib is not None:
        edge_attr = torch.stack(edge_attr)

    # from a list of numpy arrays to a torch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).T   # now we need to transpose this for compatibility sake

    #print(x.shape)    # (number of atoms, number of features)
    #print(edge_index.shape)    # (2, number of edges * 2), each edge is listed twice (i,j and j,i)
    #print(edge_attr.shape)    # (number of edges * 2, size of one-hot encoding for edge embeddings, if applicable

    if edge_attrib is not None:
        data = Data(x=x, edge_index=edge_index, edge_attr = edge_attr, y=y)     # this is a torch-geometric dataset format, compatible with NNs syntax
    else:
        data = Data(x=x, edge_index=edge_index, y=y)
    data.idx = torch.tensor([mol_id])     # keep track of mol_id, since later shuffling
    
    return data

    

class FlexibleGNNLayer(nn.Module):   #base class for all neural network components in pytorch

    """Flexible GNN layer where user can choose whether to use just node or even edge message passing """
    
    def __init__(self, node_in_dim, out_dim, edge_in_dim = None):   # constructor
        super().__init__()

        self.linear_node = nn.Linear(node_in_dim, out_dim) # The Linear layer is our learnable function that updates node embeddings after aggregating messages. # e' la classica y = xW.T + b delle reti fully connected

        if edge_in_dim is not None:   # if size of edge attributes is passed, then we want to use that also
            self.linear_edge = nn.Linear(edge_in_dim, out_dim) 
        else:
            self.linear_edge = None
            
    def forward(self, x, edge_index, edge_attr=None):   # this function MUST be overridded everytime
        """
        x: [num_nodes, node_in_dim]
        edge_index: [num_edges, 2], where each row is [source, target]
        edge_attr: [num_edges, edge_in_dim], edge embedding

        This is the key method that does the **message passing** and **embedding update**.
        This details how data flows from one layer to the next
        """
        
        num_nodes = x.shape[0]
        node_messages = torch.zeros(num_nodes, x.shape[1])

        use_edges = edge_attr is not None

        if use_edges:
            print("[Info] Using edge attributes")
            edge_messages = torch.zeros(num_nodes, edge_attr.shape[1])
        else:
            print("[Info] Edge attributes not provided — using node-only message passing.")

        # NODE:message passing + basic aggregating (= sum!); just node embeddings
        for k, (src, tgt) in enumerate(edge_index):
            node_messages[tgt] += x[src]  # Add message from src to tgt: messages[i] will hold the sum of all messages (here, node embeddings) received by node i from its neighbors.
            if use_edges:
                edge_messages[tgt] += edge_attr[k]  # EDGE: message passing + sum aggregation, 
        
        """This is the UPDATE step:

        Each node has received messages from neighbors (i.e., messages[i]). 
        We transform those aggregated messages using a linear layer (i.e., a fully connected layer with weights and bias).
        This is a learnable transformation to produce updated node embeddings
        
        Please note that if both nodes and edges are propagated, their embeddings may not share the same dimensionality, so
        we shall use two independent Linear functions """
        
        out = self.linear_node(node_messages)
        if use_edges and self.linear_edge:
            out += self.linear_edge(edge_messages)
            
        out = F.relu(out)    
        return out      # Output is the updated node embeddings after this GNN layer. These can be fed to: another GNN layer, a pooling layer(to get a graph embedding), a classifier or regressor"""

# === Define Graph-level GNN model ===

class GraphClassifier(nn.Module):

    """
    This defines a PyTorch model that:
    - Takes a molecular graph as input (nodes + edges + edge_attributes (if applicable))
    - Uses 2 Flexible GNN layers to update atom (node) (OR node + edge) embeddings
    - Pools those embeddings into a single graph embedding
    - Passes that through an linear
    - Returns a scalar (to be later smoothed wiht a sigmoid to map to either 0 or 1)

    Please note that the graph connectivity is never changed here, both node and edge embeddings may be propagated, depending on input
    """
    
    def __init__(self, node_in_dim, hidden_dim, edge_in_dim = None):
        super().__init__()

        # here I instantiate the layers
        self.gnn1 = FlexibleGNNLayer(node_in_dim, hidden_dim, edge_in_dim)  # message passing + Relu(): node embedding from (in_dim) to (hidden_dim)
        self.gnn2 = FlexibleGNNLayer(hidden_dim, hidden_dim, edge_in_dim)   # message passing + Relu(): node embedding from (hidden_dim) to (hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)   # takes graph level embedding and maps it to a logit only
                                                     # which will then be sqashed to 1 or 0 by a sigmoid (classification)

                                                     # this is simple, but we can expand this as Linear -> Relu -> Linear

    def forward(self, node_feats, edge_index, edge_attr = None):   # define the architecture

        # here I call the layers
        h = self.gnn1(node_feats, edge_index, edge_attr)    # it automatically call the .forward() method behind the scene
        h = self.gnn2(h, edge_index, edge_attr)

        # Sum pooling: [num_nodes, hidden_dim] → [1, hidden_dim], from node to graph embedding, this represents the whole molecule
        graph_embedding = h.sum(dim=0, keepdim=True)          # final pooling: collect all nodes into a graph thing 
                                                            # a node classifier would not have this
        """In GNNs, after several message-passing layers, each node has its own learned embedding vector 
                                                            — a dense feature representation summarizing its local neighborhood. But often, you want a single 
                                                            vector representing the entire graph (e.g., a molecule) for tasks like classification or regression."""
        out = self.classifier(graph_embedding)  # output is [1, 1], a logit
        
        return out.squeeze(1)    # remove one of the 1 dimensions, to make calculation easier down the line
    
   

def  model_training_classifier(model, optimizer, loss_fn, train_loader, val_loader, n_epochs, device, patience, early_stop = None):

    """
    A pre-instantiated classifier is trained with a given loss function and optimizer. early_stopping is manually coded as an option, as
    training is evaluated on the fly on a validation dataset
    INPUT:
        * model (`torch` model) (instantiated)
        * optimizer (`torch` optimizer): type of gradient descent, ADAM, etc
        * loss_fn (`torch` loss function): loss function to minimize during draining, capturing the difference between input and prediction
                                            For this classifier, we use Binary Cross Entropy with Logits
        * train_loader, `torch-geometric` DataLoader: dataset for training
        * val_loader, `torch-geometric` DataLoader: dataset for validation
        * n_epochs (int): number of epochs for training
        * device
        * patience (int): number of epochs to monitor validation loss before deciding to possibly suspend training
        * early_stop: not None, if we want to use early_stopping for better training
    OUTPUT:
        * model (`torch` model) (optimized): trained model to be used, deployed, what have you
    """

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model.to(device)

    for epoch in range(n_epochs):

        model.train()
        total_train_loss = 0

        # ===== train model on the full dataset, once ======
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # for curent batch, compute current model prediction on the input and store in "labels" the actual predictions
            pred = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.float().unsqueeze(1)
            
            # compute the loss function for this batch
            loss = loss_fn(pred, labels)
            
            # backgropagate
            loss.backward()
            optimizer.step()

            # add loss from this batch to the total training loss
            total_train_loss += loss.item()

        # ======= the model has swept over all batches "epoch" times. Now, propagate forward on the validation test and test ======
        if epoch % 10 == 0 or epoch == n_epochs - 1:

            # model optimized for now, ready to propagate forward
            model.eval()
            with torch.no_grad():

                total_correct = 0
                total = 0
                total_val_loss = 0

                for val_batch in val_loader:

                    val_batch = val_batch.to(device)

                    # extract the input and output for the batch, compute the loss for the batch, add that to the full validation loss
                    pred = torch.sigmoid(model(val_batch.x, val_batch.edge_index, val_batch.batch)).squeeze(1)
                    labels = val_batch.y.view(-1)

                    val_loss = loss_fn(pred, labels)
                    total_val_loss += val_loss.item()

                    # classify: map probabilities to binary 0 and 1
                    predicted = (pred > 0.5).float().view(-1)
                    total += labels.size(0)

                    total_correct += (predicted == labels).sum().item() # count those predictions that match the known labels

                # average loss on the validation dataset
                avg_val_loss = total_val_loss / len(val_loader)
                val_acc = total_correct / total

                print(f"Epoch {epoch} | Train Loss: {total_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}")

            # === Early stopping check ===
            if early_stop is not None:

               if avg_val_loss < best_val_loss:     # if performance on validation set is still improving, keep going
                  best_val_loss = avg_val_loss
                  best_model_state = copy.deepcopy(model.state_dict())    # save model parameters
                  patience_counter = 0
               else:
                  patience_counter += 1
                  if patience_counter >= patience:    # performance on validation set stopped improving
                      print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                      break

    # Load best weights
    model.load_state_dict(best_model_state)

    return model


def model_testing(model, test_loader, device, classifier = None, regressor = None):# Set model in evaluation mode

    """
    A (regressor/classifier) model is tested on a test dataset. Returns the predicted v actual labels 
    INPUT:
        * model (`torch` model) (possibly optimized)
        * test_loader, `torch-geometric` DataLoader: test dataset
        * device
        * classifier/regressor: flag, specify which type of task the model is performing
    OUTPUT:
        * y_pred ((n_samples) np array): predicted labels
        * y_true ((n_samples) np array): actual labels
    """
    
    model.eval()

    all_preds = []
    all_labels = []

    if classifier is not None:
    
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch))
                predicted = (pred > 0.5).float().view(-1)

                all_preds.append(predicted)
                all_labels.append(batch.y)

    if regressor is not None:
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # extract features ('X') and labels ('y')
                all_preds.append(model(batch.x, batch.edge_index, batch.batch))
                all_labels.append(batch.y.float())
                
                
    # now we are going to use some scikit-learn functions that only take in NumPy arrays. We need to convert tensors then.
    y_pred = torch.cat(all_preds, dim=0).cpu().numpy()
    y_true = torch.cat(all_labels, dim = 0).cpu().numpy()

    return y_pred, y_true
    
    
