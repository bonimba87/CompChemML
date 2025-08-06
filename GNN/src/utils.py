#=========================================================================================
# Set of helper functions or modules that are used in notebooks using Graph Neural Networks
#=========================================================================================

import numpy as np

import torch.nn as nn
import torch.nn.functional as F   # these are the activation functions
from torch.utils.data import random_split

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader


def split_data(full_data, train_ratio, val_ratio):

    """
    Split a PyTorch dataset into train, validation, and test subsets.

    Parameters
    ----------
    full_data : torch.utils.data.Dataset
        The full dataset containing molecular graphs as torch_geometric Data objects.
    train_ratio : float
        Fraction of the full dataset to allocate for training.
    val_ratio : float
        Fraction of the full dataset to allocate for validation.

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        Training portion of the dataset.
    val_dataset : torch.utils.data.Subset
        Validation portion of the dataset.
    test_dataset : torch.utils.data.Subset
        Remaining portion of the dataset used for testing.

    Notes
    -----
    The function uses `random_split()` and ensures that rounding errors are handled
    by allocating the remainder to the test set.
    """

    total_size = len(full_data)
    train_size = int(train_ratio * total_size)
    val_size   = int(val_ratio * total_size)
    test_size  = total_size - train_size - val_size  # handles rounding

    train_dataset, test_dataset, val_dataset = random_split(full_data, [train_size, test_size, val_size])

    return train_dataset, val_dataset, test_dataset



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
    Extract atom-level features to be used as node attributes in a molecular graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit Atom object, typically obtained from `mol.GetAtoms()`.

    Returns
    -------
    torch.Tensor
        A 1D tensor of floats containing:
        - Atomic number
        - Degree (number of directly bonded neighbors)
        - Implicit valence
        - Formal charge
        - Aromaticity flag (0 or 1)

    Notes
    -----
    This feature vector provides basic topological and electronic information about the atom.
    It is used as input to the GNN at each node.
    """
    
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())], dtype=torch.float)

def bond_features(bond):

    """
    Extract bond-level features to be used as edge attributes in a molecular graph.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit Bond object, typically obtained from `mol.GetBonds()`.

    Returns
    -------
    torch.Tensor
        A 1D tensor of floats (length 4), representing a one-hot encoding of the bond type:
        - [1, 0, 0, 0] → Single bond
        - [0, 1, 0, 0] → Double bond
        - [0, 0, 1, 0] → Triple bond
        - [0, 0, 0, 1] → Aromatic bond

    Notes
    -----
    This representation is a simple one-hot encoding of bond types and does not currently 
    include stereochemistry, ring status, or bond conjugation. Extend as needed.
    """

    bt = bond.GetBondType()
    return torch.tensor([int(bt == Chem.rdchem.BondType.SINGLE),
                    int(bt == Chem.rdchem.BondType.DOUBLE),
                    int(bt == Chem.rdchem.BondType.TRIPLE),
                    int(bt == Chem.rdchem.BondType.AROMATIC)], dtype=torch.float)

def graph_featurizer_pygeom(mol, mol_id, y, edge_attrib = None):

    """
    Converts an RDKit molecule into a PyTorch Geometric `Data` object with atomic and bond features.

    This function translates a molecule into a graph representation, where atoms are nodes 
    and bonds are edges. Atom features are computed using a user-defined `atom_features` 
    function, and optionally bond (edge) features using `bond_features`. The resulting 
    graph is undirected, so each bond is represented in both directions.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object to be converted into a graph.
    
    mol_id : int
        Unique identifier for the molecule. Stored as `data.idx` to enable tracking 
        during shuffling or batching.

    y : torch.Tensor
        Target label(s) associated with the molecule. Used for supervised learning tasks.

    edge_attrib : optional
        If not None, edge (bond) features will be computed and included in the `Data` object.
        This must be any non-None value (e.g., True) to activate bond feature extraction.

    Returns
    -------
    data : torch_geometric.data.Data
        A PyTorch Geometric `Data` object with the following attributes:
            - x: node feature matrix of shape (num_atoms, num_node_features)
            - edge_index: edge list tensor of shape (2, num_edges)
            - edge_attr (optional): edge feature matrix if `edge_attrib` is not None
            - y: target label(s) for the graph
            - idx: tensor containing `mol_id` for tracking
    
    Notes
    -----
    - The graph is undirected: each bond is added twice (i->j and j->i).
    - The user must define `atom_features(atom)` and optionally `bond_features(bond)` 
      to extract atom and bond-level features as torch tensors.
    - Designed for compatibility with GNN models in PyTorch Geometric.
    """

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
        if len(edge_attr) != 0:
            edge_attr = torch.stack(edge_attr)
        else:
            edge_attr = torch.tensor([])

    # from a list of numpy arrays to a torch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).T   # now we need to transpose this for compatibility sake

    if edge_attrib is not None:
        data = Data(x=x, edge_index=edge_index, edge_attr = edge_attr, y=y)     # this is a torch-geometric dataset format, compatible with NNs syntax
    else:
        data = Data(x=x, edge_index=edge_index, y=y)
    data.idx = torch.tensor([mol_id])     # keep track of mol_id, since later shuffling
    
    return data

    

class FlexibleGNNLayer(nn.Module):   #base class for all neural network components in pytorch

    """
    A flexible GNN message passing layer that supports both node-only and edge-aware propagation.

    This layer performs:
    - Neighborhood aggregation via simple sum over neighbor messages
    - Optional use of edge features to enrich the message
    - Node embedding update via linear transformation and ReLU activation

    Parameters
    ----------
    node_in_dim : int
        Dimension of input node features.
    out_dim : int
        Dimension of output node embeddings.
    edge_in_dim : int or None, optional
        Dimension of edge features. If None, edge features are not used in message passing.

    Forward Inputs
    --------------
    x : torch.Tensor
        Node feature matrix of shape [num_nodes, node_in_dim].
    edge_index : torch.Tensor
        Tensor of shape [num_edges, 2], where each row represents [source_node, target_node] in COO format.
    edge_attr : torch.Tensor or None, optional
        Edge feature matrix of shape [num_edges, edge_in_dim]. Required only if `edge_in_dim` is provided.

    Returns
    -------
    out : torch.Tensor
        Updated node embeddings of shape [num_nodes, out_dim].

    Notes
    -----
    - Message passing is done via simple summation from neighboring nodes.
    - If `edge_attr` is provided, edge messages are added to node messages before transformation.
    - Linear layers for nodes and edges are learned independently.
    - Activation is ReLU.
    - This layer does not change graph connectivity or edge structure.
    """
    
    
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
            #print("[Info] Using edge attributes")
            edge_messages = torch.zeros(num_nodes, edge_attr.shape[1])
        else:
            #print("[Info] Edge attributes not provided — using node-only message passing.")

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
    Graph Neural Network for binary classification of molecular graphs.

    This model consists of two message-passing layers (via FlexibleGNNLayer),
    followed by a sum pooling operation to produce a graph-level embedding,
    and a final linear layer to predict a logit (used for binary classification).

    Parameters
    ----------
    node_in_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden (and output) node embeddings.
    edge_in_dim : int or None, optional
        Dimension of edge features. If None, edge features are ignored.

    Forward Inputs
    --------------
    node_feats : torch.Tensor
        Node feature matrix of shape [num_nodes, node_in_dim].
    edge_index : torch.Tensor
        Graph connectivity with shape [2, num_edges].
    edge_attr : torch.Tensor or None, optional
        Edge features matrix of shape [num_edges, edge_in_dim] (if applicable).

    Returns
    -------
    out : torch.Tensor
        Scalar output (logit) predicting the class of the entire graph.

    Notes
    -----
    - This is a graph-level model; node embeddings are aggregated via sum pooling.
    - Final output is a single logit to be passed through `torch.sigmoid` externally.
    - Edge features are supported if `edge_in_dim` is provided.
    - Graph connectivity is never changed across a GNN layer
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
    Train a binary classifier on graph data using PyTorch, with optional early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated PyTorch model (e.g., GNN-based classifier).
    optimizer : torch.optim.Optimizer
        Optimizer to update model weights (e.g., Adam).
    loss_fn : torch.nn.Module
        Loss function to minimize, typically `BCEWithLogitsLoss` for binary classification.
    train_loader : torch_geometric.data.DataLoader
        Dataloader for training dataset.
    val_loader : torch_geometric.data.DataLoader
        Dataloader for validation dataset.
    n_epochs : int
        Number of training epochs.
    device : torch.device
        Computation device ('cpu' or 'cuda').
    patience : int
        Number of epochs to wait for improvement before triggering early stopping.
    early_stop : bool or None, optional
        If not `None`, activates early stopping logic.

    Returns
    -------
    model : torch.nn.Module
        The trained model with parameters from the best epoch (lowest validation loss).

    Notes
    -----
    - Evaluates the model on the validation set every 10 epochs (and final epoch).
    - Applies sigmoid activation and 0.5 thresholding for binary classification.
    - Keeps track of the best model state using validation loss for early stopping.
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
    Evaluate a trained model on a test dataset for either classification or regression.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model to evaluate.
    test_loader : torch_geometric.data.DataLoader
        DataLoader containing the test set.
    device : torch.device
        Computation device ('cpu' or 'cuda').
    classifier : bool, optional
        Set to True if evaluating a classification model. Applies sigmoid activation.
    regressor : bool, optional
        Set to True if evaluating a regression model.

    Returns
    -------
    y_pred : np.ndarray
        Array of model predictions, shape (n_samples,).
    y_true : np.ndarray
        Array of true labels, shape (n_samples,).

    Notes
    -----
    - This function assumes a binary classification task for `classifier=True`.
    - Predictions are thresholded at 0.5 for binary classification.
    - Predictions and ground truths are returned as NumPy arrays for metric evaluation.
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
    
    
