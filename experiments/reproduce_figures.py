"""Regenerate every committed figure from committed logs and checkpoints.

No training. Training this repo end to end is roughly three hours of CPU --
the Burgers run alone is ~30 minutes of Adam, and the spectral-bias sweep is
twelve PINN solves -- so the artifacts ship with the repo and this script turns
them back into the figures:

    logs/*.csv                  -> heat_convergence, spectral_regression,
                                   spectral_pinn, optimizer_study,
                                   crank_nicolson, inverse, highd_mesh,
                                   highd_pinn, highd_crossover
    logs + checkpoints/*.pt     -> spectral_fix, hard_bc
    checkpoints/*.pt            -> heat_error, burgers_error, burgers_slices,
                                   adaptive_collocation

The split is not arbitrary: a figure of a *curve* can come from a CSV, but a
figure of a *field* (an error heatmap, a profile, a slice) is a picture of the
trained network itself, and only the weights can reproduce that. See
``pinn/checkpoints.py`` for the format and why the constructor config travels
with the state dict.

Artifacts are checked for presence first, so a missing or renamed file fails
loudly here rather than deep inside a plotting call, and
``tests/test_reproduce_figures.py`` asserts that FIGURES matches the contents
of ``figures/`` in both directions -- a new figure without a replay path fails
the suite instead of silently becoming unreproducible.

Run:  python experiments/reproduce_figures.py
"""

import sys
from pathlib import Path

import adaptive_collocation
import burgers
import crank_nicolson
import hard_bc
import heat
import highd_crossover
import highd_mesh
import highd_pinn
import inverse
import loss_weighting
import optimizer_study
import spectral_bias

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
CKPTS = ROOT / "checkpoints"

# Every figure this script regenerates, i.e. every figure the repo ships.
FIGURES = (
    "heat_error.png",             # heat, from the checkpoint
    "heat_convergence.png",       # heat, from the sweep CSVs
    "burgers_error.png",          # burgers, from the checkpoint
    "burgers_slices.png",         # burgers, from the checkpoint
    "spectral_regression.png",    # spectral_bias, from CSV
    "spectral_pinn.png",          # spectral_bias, from CSV
    "spectral_fix.png",           # spectral_bias, CSV + the two k=32 checkpoints
    "optimizer_study.png",        # optimizer_study, from CSVs
    "crank_nicolson.png",         # crank_nicolson, from CSV
    "adaptive_collocation.png",   # adaptive_collocation, from 3 checkpoints
    "hard_bc.png",                # hard_bc, from CSV + the ansatz checkpoint
    "inverse.png",                # inverse, from the two sweep CSVs
    "loss_weighting.png",         # loss_weighting, from its two CSVs
    "highd_mesh.png",             # highd_mesh, from its two cost CSVs
    "highd_pinn.png",             # highd_pinn, from the sweep/trace/cost CSVs
    "highd_crossover.png",        # highd_crossover, from its mesh/pinn/probe CSVs
)

LOG_FILES = (
    "heat_collocation.csv", "heat_width.csv",
    "spectral_regression.csv", "spectral_pinn.csv", "spectral_long.csv",
    "optimizer_adam.csv", "optimizer_lbfgs.csv", "optimizer_hybrid.csv",
    "crank_nicolson.csv", "hard_bc.csv", "adaptive_collocation.csv",
    "inverse.csv", "inverse_trace.csv",
    "loss_weighting.csv", "loss_weighting_trace.csv",
    "highd_mesh_cost.csv", "highd_mesh_extrapolated.csv",
    "highd_mesh_order.csv", "highd_mesh_flat.csv", "highd_mesh_steps.csv",
    "highd_pinn_sweep.csv", "highd_pinn_trace.csv", "highd_pinn_cost.csv",
    "highd_crossover_mesh.csv", "highd_crossover_pinn.csv",
    "highd_crossover_probe.csv", "highd_crossover_probe_trace.csv",
    "highd_crossover_summary.csv",
)

CKPT_FILES = (
    heat.CKPT,
    burgers.CKPT,
    hard_bc.CKPT,
    spectral_bias.ckpt_name("plain", max(spectral_bias.K_VALUES)),
    spectral_bias.ckpt_name("fourier", max(spectral_bias.K_VALUES)),
) + tuple(adaptive_collocation.ckpt_name(m) for m in adaptive_collocation.ARMS)


def check_artifacts():
    """Return the list of missing committed files the figures need (empty if OK)."""
    missing = [f"logs/{n}" for n in LOG_FILES if not (LOGS / n).exists()]
    missing += [f"checkpoints/{n}" for n in CKPT_FILES if not (CKPTS / n).exists()]
    return missing


def main():
    missing = check_artifacts()
    if missing:
        print("ERROR: missing committed artifacts required to reproduce figures:")
        for m in missing:
            print(f"  - {m}")
        print("Re-run the experiment that produces them (see README 'Reproduce').")
        return 1

    for label, fn in (
        ("heat equation (error field + convergence)", heat.figures_from_committed),
        ("Burgers (error field + slices)", burgers.figures_from_committed),
        ("spectral bias (regression, sweep, failure/fix)",
         spectral_bias.figures_from_committed),
        ("optimizer study (Adam vs L-BFGS)", optimizer_study.make_figure),
        ("Crank-Nicolson baseline", crank_nicolson.figures_from_committed),
        ("adaptive collocation (RAD vs RAR)",
         adaptive_collocation.figures_from_committed),
        ("hard boundary conditions", hard_bc.figures_from_committed),
        ("inverse problem (alpha recovery)", inverse.figures_from_committed),
        ("loss weighting (residual vs IC/BC balance)",
         loss_weighting.figures_from_committed),
        ("high-d mesh baseline (cost to fixed accuracy vs d)",
         highd_mesh.figures_from_committed),
        ("high-d PINN sweep (error and cost vs d)",
         highd_pinn.figures_from_committed),
        ("high-d crossover (mesh vs PINN at equal accuracy)",
         highd_crossover.figures_from_committed),
    ):
        print(f"\n>>> {label}")
        fn()

    print(f"\nAll {len(FIGURES)} figures reproduced into figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
