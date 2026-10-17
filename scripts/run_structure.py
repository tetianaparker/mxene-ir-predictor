"""Γ-point IR prediction for one surface-terminated MXene (see CLAUDE.md).

Usage: python scripts/run_structure.py <name> [--potential mace|mpa0|chgnet] [--relax-only]

Builds the slab from data/structures.csv, relaxes in-plane lattice + ions with
MACE-MP-0 or CHGNet (c fixed), runs phonopy finite displacements (2x2x1, 0.01 A),
and writes runs/<name>/summary.json plus rows in results/predictions.csv
(mpa0 = MACE-MPA-0 medium, CHGNet: runs/<name>_<potential>/, results/predictions_<potential>.csv).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parents[1]
THZ_TO_CM1 = 33.35641
SUPERCELL = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
DISP = 0.01
FMAX = 1e-4  # eV/A; phonons need a tight minimum
IMAG_TOL = -5.0  # cm-1; below this an imaginary mode is flagged as real, not noise

# Hexagonal in-plane sites (fractional) and initial interlayer guesses (A)
SITES = {"A": (0.0, 0.0), "B": (1 / 3, 2 / 3), "C": (2 / 3, 1 / 3)}
DZ_MX = 1.24  # M-X layer spacing, rocksalt (111) estimate
BOND_MT = 2.0  # M-termination bond length guess
OH_BOND = 0.97


def load_row(name):
    with open(ROOT / "data/structures.csv") as f:
        for row in csv.DictReader(f):
            if row["name"] == name:
                return row
    sys.exit(f"{name} not in data/structures.csv")


def build_slab(row):
    """M_n X_{n-1} T_2 slab with rocksalt-like ABC stacking, symmetric terminations.

    Layers alternate M/X; M uses A,B,C,... and X sits in the octahedral site between
    adjacent M layers. 'fcc' termination sits over the site with no atom in the two
    layers beneath (i.e. above the second M layer); 'hcp' sits above the X beneath.
    """
    M, n, X, term, site = row["M"], int(row["n"]), row["X"], row["termination"], row["site"]
    a = float(row["a_A"])
    vacuum = float(row["vacuum_A"])
    xs = [X] * (n - 1) if X != "CN" else ["C", "N"][: n - 1]  # mixed X: one layer each
    order = ["A", "B", "C"]
    layers = []  # (symbol, site, z)
    z = 0.0
    for i in range(n):
        m_site = order[i % 3]
        layers.append((M, m_site, z))
        if i < n - 1:
            x_site = order[(i + 2) % 3]  # the site not used by M_i or M_{i+1}
            layers.append((xs[i], x_site, z + DZ_MX))
            z += 2 * DZ_MX

    def term_layers(top):
        m1, x1 = (layers[-1], layers[-2]) if top else (layers[0], layers[1])
        if site == "fcc":
            s = ({"A", "B", "C"} - {m1[1], x1[1]}).pop()
        elif site == "hcp":
            s = x1[1]
        else:
            sys.exit(f"unknown site {site}")
        h = np.sqrt(max(BOND_MT**2 - (a / np.sqrt(3)) ** 2, 0.8**2))
        sign = 1 if top else -1
        zt = m1[2] + sign * h
        if term == "OH":
            return [("O", s, zt), ("H", s, zt + sign * OH_BOND)]
        return [(term, s, zt)]

    layers = term_layers(False) + layers + term_layers(True)
    zs = np.array([l[2] for l in layers])
    thickness = zs.max() - zs.min()
    c = thickness + vacuum
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    scaled = [[*SITES[s], (zz - zs.min() + vacuum / 2) / c] for _, s, zz in layers]
    atoms = Atoms([l[0] for l in layers], scaled_positions=scaled, cell=cell, pbc=True)
    return atoms, thickness


def slab_thickness(atoms):
    z = atoms.positions[:, 2]
    return z.max() - z.min()


def geometry_report(atoms):
    lines = [f"cell a={atoms.cell.lengths()[0]:.4f} c={atoms.cell.lengths()[2]:.3f} A, "
             f"thickness={slab_thickness(atoms):.3f} A, formula {atoms.get_chemical_formula()}"]
    for s, p, f in zip(atoms.get_chemical_symbols(), atoms.positions, atoms.get_scaled_positions()):
        lines.append(f"  {s:2s} frac=({f[0]:.4f},{f[1]:.4f},{f[2]:.4f})  z={p[2]:8.4f}")
    return "\n".join(lines)


def get_calc(potential, model):
    if potential == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        m = CHGNet.load()
        return CHGNetCalculator(model=m, use_device="cpu"), f"CHGNet {m.version}"
    from mace.calculators import mace_mp
    if potential == "mpa0":
        return mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64"), "MACE-MPA-0 medium"
    return mace_mp(model=model, device="cpu", default_dtype="float64"), f"MACE-MP-0 {model}"


def outputs(potential):
    """Paths for predictions CSV and committed summaries, per potential."""
    if potential != "mace":
        return (ROOT / f"results/predictions_{potential}.csv", ROOT / f"results/summaries_{potential}",
                f"_{potential}")
    return ROOT / "results/predictions.csv", ROOT / "results/summaries", ""


def relax(atoms, calc, logfile):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    # Voigt mask (xx, yy, zz, yz, xz, xy): in-plane cell only, c fixed
    filt = FrechetCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1])
    opt = BFGS(filt, logfile=str(logfile))
    converged = opt.run(fmax=FMAX, steps=1000)
    atoms.set_constraint()
    return atoms, bool(converged), opt.nsteps


def to_phonopy(atoms):
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:],
                        scaled_positions=atoms.get_scaled_positions())


def ir_active(characters, rotations):
    """Vector rep (x,y,z) has character tr(R); IR active iff it overlaps the mode's rep."""
    chi_v = np.array([np.trace(r) for r in rotations])
    return float(np.real(np.vdot(characters, chi_v)) / len(rotations)) > 0.5


def phonons(atoms, calc):
    ph = Phonopy(to_phonopy(atoms), supercell_matrix=SUPERCELL, primitive_matrix=np.eye(3))
    ph.generate_displacements(distance=DISP, is_plusminus=True)
    forces = []
    for sc in ph.supercells_with_displacements:
        s = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        s.calc = calc
        forces.append(s.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    fc_raw = ph.force_constants.copy()
    ph.symmetrize_force_constants()  # enforces translational invariance (ASR) + index symmetry
    asr_correction = float(np.abs(ph.force_constants - fc_raw).max())

    ph.set_irreps([0, 0, 0], degeneracy_tolerance=1e-3)
    irr = ph.irreps
    freqs_cm = irr.frequencies * THZ_TO_CM1
    modes = []
    for k, bands in enumerate(irr.band_indices):
        label = irr._ir_labels[k] if irr._ir_labels is not None else None
        active = ir_active(irr.characters[k], irr.conventional_rotations)
        f = float(np.mean(freqs_cm[bands]))
        modes.append({"bands": [int(b) for b in bands], "frequency_cm1": round(f, 2),
                      "degeneracy": len(bands), "irrep": label, "ir_active": active})
    # three lowest bands are the acoustic modes at Gamma
    acoustic = set(np.argsort(np.abs(freqs_cm))[:3].tolist())
    for m in modes:
        m["acoustic"] = all(b in acoustic for b in m["bands"])
        m["imaginary"] = m["frequency_cm1"] < IMAG_TOL and not m["acoustic"]
    acoustic_max = float(np.abs(freqs_cm[sorted(acoustic)]).max())
    return ph, modes, asr_correction, acoustic_max, irr._pointgroup_symbol, len(forces)


FIELDS = ["name", "termination", "frequency_cm1", "irrep", "ir_active", "reliable"]


def append_predictions(path, name, term, modes, reliable):
    path.parent.mkdir(exist_ok=True)
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != name]  # rerun replaces
    for m in modes:
        if m["acoustic"]:
            continue
        rows.append({"name": name, "termination": term, "frequency_cm1": m["frequency_cm1"],
                     "irrep": m["irrep"], "ir_active": m["ir_active"], "reliable": reliable})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--potential", choices=["mace", "mpa0", "chgnet"], default="mace")
    ap.add_argument("--model", default="medium", help="MACE-MP-0 size")
    ap.add_argument("--site", help="override the site column (e.g. for a site check)")
    ap.add_argument("--relax-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    row = load_row(args.name)
    if args.site:
        row["site"] = args.site
    pred_csv, summ_dir, suffix = outputs(args.potential)
    run = ROOT / "runs" / (args.name + suffix + args.tag)
    run.mkdir(parents=True, exist_ok=True)

    atoms, _ = build_slab(row)
    write(run / "initial.vasp", atoms, format="vasp", direct=True, sort=False)
    print("== initial geometry ==\n" + geometry_report(atoms), flush=True)

    calc, model_label = get_calc(args.potential, args.model)
    relaxed, converged, nsteps = relax(atoms, calc, run / "relax.log")
    relaxed.calc = calc
    energy = float(relaxed.get_potential_energy())
    fmax = float(np.abs(relaxed.get_forces()).max())
    write(run / "relaxed.vasp", relaxed, format="vasp", direct=True, sort=False)
    a0, a1 = atoms.cell.lengths()[0], relaxed.cell.lengths()[0]
    print(f"== relaxed geometry (converged={converged}, steps={nsteps}, E={energy:.5f} eV, "
          f"max|F|={fmax:.2e}) ==\n" + geometry_report(relaxed), flush=True)
    print(f"a: csv {float(row['a_A']):.4f} -> relaxed {a1:.4f} A ({100 * (a1 / a0 - 1):+.2f}%)", flush=True)

    summary = {
        "name": args.name, "row": row, "model": model_label,
        "settings": {"supercell": "2x2x1", "vacuum_A": float(row["vacuum_A"]), "displacement_A": DISP,
                     "relax_fmax_eV_A": FMAX, "c_fixed_A": round(float(atoms.cell.lengths()[2]), 4)},
        "a_input_A": round(float(a0), 4), "a_relaxed_A": round(float(a1), 4),
        "thickness_relaxed_A": round(float(slab_thickness(relaxed)), 4),
        "relax_converged": converged, "relax_steps": nsteps, "energy_eV": energy,
        "energy_per_atom_eV": energy / len(relaxed),
    }
    if not args.relax_only:
        ph, modes, asr_corr, ac_max, pg, ndisp = phonons(relaxed, calc)
        ph.save(run / "phonopy_params.yaml", settings={"force_constants": True})
        summary.update({
            "point_group": pg, "n_displacements": ndisp,
            "asr_max_fc_correction_eV_A2": asr_corr, "acoustic_max_abs_cm1": round(ac_max, 2),
            "imaginary_modes": [m for m in modes if m["imaginary"]], "modes": modes,
        })
    # a manual reliable=false verdict (with its note) survives reruns until edited by hand
    summ_path = summ_dir / f"{args.name}.json"
    old = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summary["reliable"] = old.get("reliable", True)
    if "reliability_note" in old:
        summary["reliability_note"] = old["reliability_note"]
    if not args.relax_only and not args.tag:
        append_predictions(pred_csv, args.name, row["termination"], modes, summary["reliable"])
    text = json.dumps(summary, indent=2)
    (run / "summary.json").write_text(text)
    if not args.tag and not args.relax_only:  # runs/ is gitignored; keep a committed copy
        summ_dir.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(text)
    print(f"wrote {run / 'summary.json'}")


if __name__ == "__main__":
    main()
