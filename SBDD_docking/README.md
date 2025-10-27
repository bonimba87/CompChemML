## Overview

This project is part of my cheminformatics and computational-drug-discovery upskilling portfolio. It focuses on implementing a practical docking workflow from scratch using open-source tools, bridging structure-based modeling with ML-ready datasets.

## Docking in the Drug Discovery Pipeline  

Molecular docking is a **structure-based** (i.e., using the geometry and chemistry of the binding site to guide predictions, opposite to **ligand-based**)  **virtual screening (SBVS)** approach that predicts how small molecules fit into a protein’s binding site.  

- **Where it fits:** Docking is typically applied in the *hit discovery* phase, after a target has been identified but before experimental screening.  
- **What it does:**  
  - Predicts binding poses (3D orientations of ligands in the pocket).  
  - Estimates binding affinity to rank and prioritize compounds.  
- **Why it matters:** Docking can reduce millions of candidate molecules to a manageable set of promising hits, saving time and cost before lab testing.  

**Takeaway:** Docking is a fast, cost-effective filter to guide medicinal chemistry and accelerate early drug discovery.  


## Docking Workflow with `AutoDock Vina`

This repository demonstrates an **end-to-end docking pipeline** using [AutoDock Vina](http://vina.scripps.edu/).  
It covers receptor preparation, ligand conformer generation, docking, and evaluation on actives vs decoys. For the sake of dataset size, the raw `pdbqt` poses produced by the docking engines are not uploaded.

---

## Dataset

**Target protein**  
We used the X-ray crystal structure of [HIV protease (PDB: 1HVR)](https://www.rcsb.org/structure/1HVR) as the receptor.  
- Co-crystallized ligand provides a reference pose for redocking validation.  
- Non-standard residue **CSO (S-hydroxycysteine)** was retained to preserve protein integrity.  
- Waters and other irrelevant heteroatoms were removed.

**Ligands**  
- **Actives:** Known binders extracted from [**ChEMBL**](https://www.ebi.ac.uk/chembl/) (ligand IDs starting with `CHEM`).  
- **Decoys:** Structurally similar but inactive molecules used as negative controls.  

**Dataset size**  
- Actives: 15 compounds  
- Decoys: 15 compounds 

----

## Workflow Overview


1. **Receptor preparation**  
   - Fix missing atoms/residues (`pdbfixer`)  
   - Add hydrogens and Gasteiger charges  
   - Convert to Vina-compatible `.pdbqt`

2. **Ligand preparation**  
   - Input: SMILES strings  
   - Generate 3D conformers (`RDKit`)  
   - Optimize geometries (UFF)  
   - Convert to `.pdbqt` with hydrogens and charges (`Open Babel`)

3. **Docking with Vina**  
   - Rigid receptor, flexible ligands  
   - Define docking box based on co-crystal ligand  

4. **Analysis**  
   - Redocking of co-crystal ligand (RMSD check)  
   - Benchmark actives vs decoys (histograms, ROC curve, AUC ~0.77)  
   - Visualization in `ChimeraX`

---

## Results
- **Redocking**: RMSD = 0.55 Å (excellent reproduction of crystal pose)
  <p align="center">
  <img src="SBDD_docking/results/figures/comparison_docking.png" width="450"/>
  <br>
  <em> Superposition of the crystal ligand (blue) with its redocked alias (magenta). </em>
</p>

- **Actives vs Decoys**: ROC AUC = 0.77
  <p align="center">
  <img src="SBDD_docking/results/figures/auc_roc.pdf" width="450"/>
  <br>
  <em> ROC curve with an AUC value of 0.77, indicating that docking does discriminate effectively between actives and decoys. </em>
</p>

- Docking successfully enriches actives among top-ranked ligands.
- These results confirm that the pipeline can successfully reproduce experimental binding modes and distinguish actives from decoys, validating its use as a baseline SBVS workflow.

---

## Next Steps
- Scale docking to larger ligand libraries  
- Explore different scoring functions or engines (e.g., Glide, MOE)  
- Add enrichment factor metrics (EF1/5/10%)  

