# MXene IR predictor – rules for Claude Code

## Goal
Predict IR-active vibrational frequencies of surface-terminated MXenes without DFT: build the structure, get forces from a universal ML potential (MACE-MP-0 first; CHGNet if its weights won't download), run phonopy finite displacements, report Γ-point modes with irreps and IR activity, then compare with the owner's measured FTIR peaks in data/measured_peaks.csv.

## Environment
pip install ase pymatgen phonopy mace-torch and torch (CPU wheel from download.pytorch.org/whl/cpu; if that index is blocked, plain pip install torch). Verify the MACE-MP-0 weights download; otherwise pip install chgnet and use it. CPU only. Supercell 2×2×1, vacuum 20 Å along c, displacement 0.01 Å. One structure per run; everything logged under runs/<name>/.

## Workflow per structure
1. Build the cell with ASE from its row in data/structures.csv (M, n, X, termination, site, a, c). Report the geometry before running.
2. Relax the in-plane lattice and ions with the potential (c fixed); report input and relaxed a.
3. phonopy displacements → forces from the potential → force constants → Γ-point frequencies, irreps, IR-active flag.
4. Apply the acoustic sum rule; flag any imaginary modes that remain and never drop them silently.
5. Write runs/<name>/summary.json and append to results/predictions.csv (name, termination, frequency_cm1, irrep, ir_active).

## Evaluation
results/compare.csv: each measured band matched to the nearest IR-active predicted mode of the same assignment type, error in cm⁻¹, MAE per termination. Stage 2: leave-one-out linear correction measured = a·predicted + b; report MAE before and after.

## Rules
Stop and show the first Ti3C2O2 result before any other structure. Commit code, CSVs, summaries and figures only; runs/ and model weights are gitignored. Use the owner's measured peaks only from data/measured_peaks.csv; cite the source paper for each row in the write-up.
