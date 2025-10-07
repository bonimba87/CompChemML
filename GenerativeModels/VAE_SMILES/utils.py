import numpy as np, pandas as pd, os, math, random, time
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def define_vocabulary():
    
    vocab = [
    # Special tokens [padding, start, end, unknown symbol]
    "<PAD>", "<START>", "<END>", "<UNK>",
    # Common atoms
    "C", "O", "N", "S", "P", "F", "Cl", "Br", "I",
    # Aromatic atoms (lowercase form often used in SMILES)
    "c", "o", "n", "s", "p",
    # Bonds
    "-", "=", "#",
    # Branching / rings
    "(", ")", "[", "]",
    # Ring closure digits
    "1","2","3","4","5","6","7","8","9",
    # Stereo / charges
    "@", "@@", "+", "H"
     ]

    # Build dictionaries: use increasing integer numbers for tokenization
    stoi = {tok: i for i, tok in enumerate(vocab)}   # from tokens to integers
    itos = {i: tok for tok, i in stoi.items()}     # from integers to tokens

    return vocab, stoi, itos


def define_tokens(smiles: str):
    
    """ Takes in a SMILES string and returns the tokenization = split string into tokens """
    
    tokens = []
    i = 0
    while i < len(smiles):
        # check two-character tokens
        if smiles[i:i+2] in ("Cl", "Br", "@@"):
            tokens.append(smiles[i:i+2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens


def encode(smiles, stoi):
    
    """ Tokenization: from tokens to sequence of integers """
    
    tokens = ["<START>"] + define_tokens(smiles) + ["<END>"]
    unk_id = stoi["<UNK>"]
    return [stoi.get(t, unk_id) for t in tokens]

def decode(id_list, itos):
    
    """ De-tokenization: from sequence of integers to tokens """   
    
    tokens = [itos[i] for i in id_list if itos[i] not in ("<PAD>", "<START>", "<END>")]
    return "".join(tokens)

def run_epoch_rnn(loader, train=True, clip=1.0):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(xb)                            # (B,T,V)
        labels = yb.reshape(-1)
        
                         # predictions from the NN        # actual target from the data to compare against
        loss = loss_fn(pred.reshape(-1, V), labels)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
        # count non-PAD targets for averaging
        ntok = (yb != pad_id).sum().item()
        total_loss += loss.item() * max(ntok, 1)
        total_tokens += max(ntok, 1)
    return total_loss / total_tokens      # this ensures fair averaging per token, regarless of seqeucne length.
                                          # wthout this, the reported loss would be biased toward short or long SMILES



@torch.no_grad()
def decode_one_smile_from_z(z, max_len=120, temperature=0.9, top_k=30):
    
    """
    Generate a single SMILES string from a latent vector `z` using the trained VAE decoder.

    Starting from the special <START> token, this function autoregressively generates
    a sequence of tokens by repeatedly feeding the growing prefix into the decoder GRU.
    At each step, the decoder predicts a probability distribution over the vocabulary 
    for the next token, conditioned on both the latent variable `z` and all tokens 
    generated so far.

    The next token is sampled from the softmax distribution of the logits (optionally
    restricted to the top-k most probable tokens). Generation stops when the <END> 
    token is produced or when `max_len` tokens have been generated.

    Args:
        z (torch.Tensor): Latent vector of shape (1, z_dim), sampled from the Gaussian prior N(0, I).
        max_len (int): Maximum number of tokens to generate
        temperature (float): Softmax temperature controlling randomness of sampling 
            (T < 1.0 → more deterministic; T > 1.0 → more diverse).
        top_k (int or None): If set, restricts sampling to the top-k most probable 
            tokens at each step (default: 30).

    Returns:
        str: A decoded SMILES string, excluding special tokens (<START>, <END>, <PAD>).

    Notes:
        - The initial hidden state of the decoder GRU is computed from `z` via a 
          learned affine transformation (`model.init_dec_hidden`).
        - This process corresponds to **ancestral sampling** from the model’s 
          generative distribution p(x | z).
        - Uses `torch.no_grad()` to disable gradient tracking during generation.
    """

    device   = z.device if z.is_cuda or z.device.type != "cpu" else next(model.parameters()).device
    # this for sure takes in a z that was sampled
    model.eval()

    # from z to the initial hidden state to feed to the decoder
    h0 = model.init_dec_hidden(z, num_layers=model.dec_rnn.num_layers,
                               dec_h=model.dec_rnn.hidden_size)

    start_id = stoi["<START>"]
    end_id = stoi["<END>"]
    V = len(itos)
    
    ids = [start_id]
    
    for _ in range(max_len):

        # define the prefix (= tokenized string generated so far)
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        
        logits = model.decode(x, h0)[:, -1, :]   # (B,V) last step
        logits = logits / max(1e-6, temperature)
       
        if top_k is not None:
            k = min(top_k, V)
            vals, idxs = torch.topk(logits, k=k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            next_id = idxs[0, torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        
        ids.append(next_id)
        if next_id == end_id:     # if <END>, stop; otherwise, feed the updates `ids` back into the GRU for the next step
            break

    return decode(ids)