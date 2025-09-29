# SMILES Generation with RNNs (GRU Baseline)

This notebook explores **generative modeling for molecules** using a recurrent neural network (RNN) with gated recurrent units (GRUs).  
The model is trained on canonicalized SMILES strings and learns to **generate new molecular structures** in an autoregressive fashion.

---

### Objectives
- Understand how **language modeling principles** apply to molecular SMILES.  
- Build a complete pipeline: tokenization → model → training → evaluation → generation.  
- Evaluate generated molecules with **validity, uniqueness, and novelty** metrics.  
- Visualize generated molecules and basic properties (logP, QED).  
- Use this as a **baseline generative approach**, with awareness of more modern models (Transformers, VAEs, diffusion).  

---

### Approach
1. **Data preparation**  
   - Canonicalized SMILES with RDKit to ensure no duplicates or leakage across train/val/test splits.  
   - Defined a custom vocabulary including special tokens (`<PAD>`, `<START>`, `<END>`).  

2. **Model architecture**  
   - Embedding layer → GRU → Linear projection to vocabulary logits.  
   - Trained with **cross-entropy loss** (ignoring `<PAD>` tokens).  
   - Model learns conditional distributions:  
     \[
     P(x_t \mid x_{<t})
     \]

3. **Training**  
   - Reported cross-entropy and perplexity across train/val/test sets.  
   - Achieved test perplexity ≈ 2.2, indicating strong capture of SMILES syntax.  

4. **Generation**  
   - Autoregressive sampling with **temperature scaling** and **top-k filtering**.  
   - RDKit used to validate generated molecules.  
   - ![Sampled molecules visualized with descriptors (MW, logP, QED)](results/Some_generate_molecules.png)  
---

### Results
- **Validity**: ~77% (fraction of generated SMILES that parse into molecules)  
- **Uniqueness**: 100% (no duplicates among valid molecules)  
- **Novelty**: 100% (no generated molecules were in the training set, after canonicalization)  

These results demonstrate that the GRU learns SMILES grammar and generates diverse, novel molecules, though not all are chemically valid.  

---

### Reflections
Working through this project gave me a much deeper understanding of:  
- How **autoregressive models** learn distributions over sequences.  
- Why GRUs improve over vanilla RNNs for longer contexts.  
- The importance of **data hygiene** in molecular ML (canonicalization, leakage checks).  
- How to evaluate generative models beyond accuracy, using **validity, uniqueness, novelty**.  

This hands-on implementation taught me more than following pre-built tutorials, since I had to debug tokenization issues, manage padding, and implement generation from scratch.  

---

### Limitations & Next Steps
- RNNs/GRUs can still struggle with **long-range dependencies** (e.g., rings, stereochemistry).  
- Modern state-of-the-art approaches use **Transformers, VAEs, or diffusion models** for molecular generation.  
- Future extensions:  
  - Conditional generation (property-guided).  
  - Descriptor distribution analysis (e.g., logP, QED) compared to training data.  
  - Exploration of **Transformer-based architectures** for improved validity.  


