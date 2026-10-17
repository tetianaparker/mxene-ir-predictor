"""Measured FTIR bands vs predicted IR-active Γ modes (CLAUDE.md, Evaluation).

Usage: python scripts/compare.py

For each prediction set (reliable MACE-MP-0 rows, reliable MACE-MPA-0 rows) every
measured band in data/measured_peaks.csv is mapped to a structure + irrep by the
rules below and matched to the nearest IR-active predicted mode of that irrep.
Stage 2 is a leave-one-out linear correction measured = a*predicted + b, fitted on
all matched bands and, separately, on the O-terminated bands only (the two OH bands
dominate the all-band fit).
Writes results/compare.csv, results/compare_mae.csv and results/compare.png.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = {"MACE-MP-0": "results/predictions.csv", "MACE-MPA-0": "results/predictions_mpa0.csv"}

# Measured MXene (mixed termination) -> modelled O-terminated slab
O_SLAB = {"Ti3C2Tx": "Ti3C2O2", "Ti3CNTx": "Ti3CNO2", "Ti2CTx": "Ti2CO2",
          "Nb2CTx": "Nb2CO2", "Mo2CTx": "Mo2CO2", "V2CTx": "V2CO2"}
ADSORBATE_BONDS = {"C-H", "C=O", "C-O"}
# Ti3CNO2 has no inversion centre (C3v): A2u -> A1, Eu -> E
C3V = {"A2u": "A1", "Eu": "E"}


def map_band(r):
    """Return (structure, irrep, note) or (None, None, reason) per the confirmed rules."""
    mx, bond, irrep = r["mxene"], r["bond"], r["irrep"]
    if mx not in O_SLAB:
        return None, None, "excluded: MXene not modelled"
    if bond in ADSORBATE_BONDS:
        return None, None, "excluded: adsorbate band, no counterpart in a slab model"
    if bond == "C-F":
        return None, None, "excluded: no C-F bond in the F-terminated slab"
    if bond == "O-H":
        if r["vibration"] == "stretching":
            return "Ti3C2OH2", "A2u", "O-H stretch of terminal OH is out-of-plane: matched as A2u (file label Eu)"
        return "Ti3C2OH2", irrep, ""
    if bond == "Ti-F":
        return "Ti3C2F2", irrep, ""
    slab = O_SLAB[mx]
    if slab == "Ti3CNO2":
        return slab, C3V[irrep], f"C3v: {irrep} -> {C3V[irrep]}"
    return slab, irrep, ""


def load_predictions(path):
    """{structure: {irrep: [freq,...]}} for reliable, IR-active, real modes; plus termination."""
    modes, term = {}, {}
    with open(ROOT / path) as f:
        for r in csv.DictReader(f):
            term[r["name"]] = r["termination"]
            if r["reliable"] != "True" or r["ir_active"] != "True" or float(r["frequency_cm1"]) <= 0:
                continue
            modes.setdefault(r["name"], {}).setdefault(r["irrep"], []).append(float(r["frequency_cm1"]))
    return modes, term


def loo_correct(pred, meas):
    """Leave-one-out fit of meas = a*pred + b; returns corrected prediction for each point."""
    out = np.empty_like(pred)
    for i in range(len(pred)):
        keep = np.arange(len(pred)) != i
        a, b = np.polyfit(pred[keep], meas[keep], 1)
        out[i] = a * pred[i] + b
    return out


def main():
    with open(ROOT / "data/measured_peaks.csv") as f:
        measured = list(csv.DictReader(f))

    rows, mae_rows = [], []
    for set_name, path in SETS.items():
        modes, term = load_predictions(path)
        matched = []
        for r in measured:
            slab, irrep, note = map_band(r)
            out = {"set": set_name, "mxene": r["mxene"], "bond": r["bond"], "vibration": r["vibration"],
                   "measured_irrep": r["irrep"], "measured_cm1": float(r["band_mid_cm1"]),
                   "band_low_cm1": r["band_low_cm1"], "band_high_cm1": r["band_high_cm1"],
                   "confidence": r["confidence"], "source": r["source"], "structure": slab or "",
                   "termination": "", "matched_irrep": irrep or "", "predicted_cm1": "",
                   "error_cm1": "", "loo_corrected_cm1": "", "loo_error_cm1": "", "loo_o_corrected_cm1": "",
                   "loo_o_error_cm1": "", "status": "matched",
                   "note": note}
            if slab is None:
                out["status"], out["note"] = note, ""
            elif slab not in modes:
                out["status"] = f"excluded: {slab} not reliable in {set_name}"
            elif irrep not in modes[slab]:
                out["status"] = f"excluded: no IR-active {irrep} mode in {slab}"
            else:
                cands = modes[slab][irrep]
                p = min(cands, key=lambda c: abs(c - out["measured_cm1"]))
                out.update(termination=term[slab], predicted_cm1=p, error_cm1=round(p - out["measured_cm1"], 1))
                matched.append(out)
            rows.append(out)

        pred = np.array([m["predicted_cm1"] for m in matched])
        meas = np.array([m["measured_cm1"] for m in matched])
        corr = loo_correct(pred, meas)
        for m, c in zip(matched, corr):
            m["loo_corrected_cm1"] = round(float(c), 1)
            m["loo_error_cm1"] = round(float(c) - m["measured_cm1"], 1)
        o_bands = [m for m in matched if m["termination"] == "O"]
        o_corr = loo_correct(np.array([m["predicted_cm1"] for m in o_bands]),
                             np.array([m["measured_cm1"] for m in o_bands]))
        for m, c in zip(o_bands, o_corr):
            m["loo_o_corrected_cm1"] = round(float(c), 1)
            m["loo_o_error_cm1"] = round(float(c) - m["measured_cm1"], 1)
        a, b = np.polyfit(pred, meas, 1)
        for t in ["O", "F", "OH", "all"]:
            sel = [m for m in matched if t == "all" or m["termination"] == t]
            if not sel:
                continue
            mae_rows.append({
                "set": set_name, "termination": t, "n_bands": len(sel),
                "mae_cm1": round(np.mean([abs(m["error_cm1"]) for m in sel]), 1),
                "mae_loo_cm1": round(np.mean([abs(m["loo_error_cm1"]) for m in sel]), 1),
                "mae_loo_o_only_cm1": (round(np.mean([abs(m["loo_o_error_cm1"]) for m in sel]), 1)
                                       if t == "O" else ""),
                "fit_all_a": round(a, 3), "fit_all_b": round(b, 1),
                "caveat": "rests on <=2 bands" if len(sel) <= 2 else "",
            })

    with open(ROOT / "results/compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "results/compare_mae.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mae_rows[0].keys()))
        w.writeheader()
        w.writerows(mae_rows)
    plot(rows, mae_rows)


def plot(rows, mae_rows):
    colors = {"O": "#2a78d6", "F": "#eb6834", "OH": "#1baf7a"}
    markers = {"O": "o", "F": "s", "OH": "^"}
    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    for ax, set_name in zip(axes, SETS):
        pts = [r for r in rows if r["set"] == set_name and r["status"] == "matched"]
        lim = (250, 4300)
        ax.plot(lim, lim, color=muted, lw=1, ls="--", zorder=1)
        for r in pts:
            t = r["termination"]
            lo, hi = float(r["band_low_cm1"]), float(r["band_high_cm1"])
            ax.plot([r["predicted_cm1"]] * 2, [lo, hi], color=colors[t], lw=2, alpha=0.5, zorder=2,
                    solid_capstyle="round")
            ax.scatter(r["predicted_cm1"], r["measured_cm1"], s=60, marker=markers[t], color=colors[t],
                       edgecolor="white", linewidth=1.5, zorder=3)
            if r["loo_o_corrected_cm1"] != "":
                ax.scatter(r["loo_o_corrected_cm1"], r["measured_cm1"], s=60, marker=markers[t],
                           facecolor="none", edgecolor=colors[t], linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ticks = [300, 500, 1000, 2000, 3000]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(color=grid, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
        ax.set_xlabel("Predicted (cm⁻¹)", color=ink)
        mae = {m["termination"]: m for m in mae_rows if m["set"] == set_name}
        lines = [f"{t:>3} n={mae[t]['n_bands']:<2} {mae[t]['mae_cm1']:>5.0f} {mae[t]['mae_loo_cm1']:>6.0f} "
                 f"{(f"{mae[t]['mae_loo_o_only_cm1']:.0f}" if mae[t]['mae_loo_o_only_cm1'] != "" else "–"):>6}" for t in ["O", "F", "OH", "all"] if t in mae]
        ax.text(0.98, 0.03, "MAE (cm⁻¹)   raw  LOO-all  LOO-O\n" + "\n".join(lines),
                transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, color=ink, family="monospace")
        ax.set_title(f"{set_name} (reliable structures)", color=ink, loc="left", fontsize=12)
    axes[0].set_ylabel("Measured band centre (cm⁻¹)", color=ink)
    handles = [plt.Line2D([], [], marker=markers[t], ls="", color=colors[t], markeredgecolor="white",
                          markersize=8, label=f"{t}-terminated") for t in colors]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=muted,
                           markersize=8, label="after LOO correction (O-only fit)"),
                plt.Line2D([], [], color=muted, lw=2, alpha=0.5, label="measured band range"),
                plt.Line2D([], [], color=muted, lw=1, ls="--", label="predicted = measured")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), labelcolor=ink)
    fig.text(0.5, 0.005, "F and OH rest on one and two bands; their MAEs are indicative only. "
             "Data: Parker et al., Chem. Mater. 2024.", ha="center", fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(ROOT / "results/compare.png", dpi=160, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
