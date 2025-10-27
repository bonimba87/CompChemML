# Setup of Relative Binding Free Energy (RBFE) Calculations with OpenFE

## Why FEP?
Free Energy Perturbation (FEP) is a rigorous statistical mechanics method to compute relative binding affinities between ligands.  
- Helps prioritize compounds in **drug discovery**.  
- Provides more accuracy than docking or scoring functions.  
- Requires careful setup (ligand mappings, force fields, charging, thermodynamic cycle, enhanced sampling, reweighing).  

In particular, RBFE (Relative Binding Free Energy) estimates relative binding free energies (ΔΔG) between two ligands, A and B, bound to the same receptor.
The computation relies on a thermodynamic cycle, which connects experimentally inaccessible quantities (binding free energy differences) to computable ones (alchemical transformations):

<p align="center">
  <img src="rbfe_thermocycle.png" width="450"/>
  <br>
  <em>Different legs of the thermodynamic cycle, showcasing how the ΔΔG is computed.</em>
</p>

---

### Executive Summary
- Demonstrates the **setup stage** of a Free Energy Perturbation workflow using [OpenFE toolkit](https://openfreeenergy.org)  
- This notebook is based on the tutorial materials found online.
- Example application: **TYK2 ligands** benchmark system.  
- Covers protein/ligand prep, atom mapping, thermodynamic cycle, protocol definition, and analysis scaffold.  
- Simulations not run here (HPC required), but workflow shows how ΔΔG would be gathered once results are available.  

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

## Limitations
- This notebook focuses on **setup and workflow design**.  
- Simulations were **not run** — they require HPC resources and longer walltimes.

 ## Next steps
  - Running the prepared transformations on an HPC cluster.  
  - Comparing ΔΔG estimates to experimental TYK2 benchmark values.  
  - Exploring multiple atom mapping strategies (Kartograf vs Lomap).  

