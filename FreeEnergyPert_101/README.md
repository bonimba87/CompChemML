# Setup of Relative Binding Free Energy (RBFE) Calculations with OpenFE

### Executive Summary
- Demonstrates the **setup stage** of a Free Energy Perturbation workflow using [OpenFE toolkit](https://openfreeenergy.org)  
- Example application: **TYK2 ligands** benchmark system.  
- Covers protein/ligand prep, atom mapping, thermodynamic cycle, protocol definition, and analysis scaffold.  
- Simulations not run here (HPC required), but workflow shows how ΔΔG would be gathered once results are available.  

---

## Why FEP?
Free Energy Perturbation is a rigorous statistical mechanics method to compute relative binding affinities between ligands.  
- Helps prioritize compounds in **drug discovery**.  
- Provides more accuracy than docking or scoring functions.  
- Requires careful setup (ligand mappings, force fields, charging, thermodynamic cycle, enhanced sampling, reweighing).  

---

## What This Notebook Covers
1. **Ligand and protein preparation**  
   - Load ligands from SDF and protein from PDB.  
   - Define `ChemicalSystem`s.  

2. **Atom mapping**  
   - Use **LomapAtomMapper** to define A→B transformations.  
   - Visualize mappings in 2D/3D.  

3. **Thermodynamic cycle**  
   - Set up both **complex** and **solvent** legs for RBFE.  

4. **Protocol definition**  
   - Define λ schedule, equilibration, and simulation settings.  

5. **Transformation & DAG execution scaffold**  
   - Show how transformations can be serialized and run via the CLI (`openfe quickrun`).  

6. **Result gathering (ΔΔG)**  
   - Demonstrate use of `gather()` to combine simulation results (if available).  
   - Show how MBAR is applied for final free energy estimates.  

---

## Limitations & Next Steps
- This notebook focuses on **setup and workflow design**.  
- Simulations were **not run** — they require HPC resources and longer walltimes.  
- Possible next steps:  
  - Running the prepared transformations on an HPC cluster.  
  - Comparing ΔΔG estimates to experimental TYK2 benchmark values.  
  - Exploring multiple atom mapping strategies (Kartograf vs Lomap).  

