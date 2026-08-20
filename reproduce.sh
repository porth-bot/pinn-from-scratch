#!/usr/bin/env bash
#
# Regenerate every figure in figures/ from the committed logs and checkpoints.
#
#     ./reproduce.sh              # tests, then all 19 figures: ~2 min
#     PYTHON=/path/to/python ./reproduce.sh
#
# NO TRAINING happens here, and that is the point. Training this repo end to
# end is roughly nine hours of CPU: the Burgers run alone is ~30 minutes of
# Adam against a Cole-Hopf ground truth, the spectral-bias sweep is twelve PINN
# solves, and each of the two high-dimensional sweeps is over two hours. So the
# artifacts ship with the repo -- CSV logs in logs/, model weights in
# checkpoints/ -- and this turns them back into the figures.
#
# Why weights and not just CSVs: a figure of a curve can come from a log, but a
# figure of a *field* (the error heatmaps, the Burgers slices, the k=32 profile)
# is a picture of the trained network, and only the weights reproduce that.
#
# To actually retrain, run the experiments directly. The README's Reproduce
# section is the full per-flag list with costs; the expensive ones are:
#     python experiments/heat.py            # ~25 min (default solve + both sweeps)
#     python experiments/burgers.py         # ~30 min
#     python experiments/spectral_bias.py   # ~2 h  (12 solves + regression + controls)
#     python experiments/adaptive_collocation.py    # ~12 min
#     python experiments/hard_bc.py         # ~14 min
#     python experiments/inverse.py         # ~40 min (inverse problem: 14 solves)
#     python experiments/loss_weighting.py  # ~50 min (weight ablation: 60 solves)
#     python experiments/highd_pinn.py --sweep      # ~2 h  (d = 1..16, 3 seeds)
#     python experiments/highd_crossover.py --probe # ~40 min (larger-budget cells)
#     python experiments/highd_hjb.py --sweep  # ~2.5 h (HJB d-sweep: 15 solves;
#                                           #  --seconds N time-boxes and resumes)
#     python experiments/highd_degrade.py --run    # ~2 h (samplers/density/seeds)
#     python experiments/highd_degrade.py --fit    # ~35 min (supervised control)
#     python experiments/wave.py --train    # ~25 min (wave equation, 2 ICs x 3 seeds)
# The cheap ones (optimizer_study, crank_nicolson, highd_mesh, the --check and
# --metric flags) are seconds to minutes each; see the README.
#
# Determinism: the replay is pure post-processing of committed files, so it is
# exact. Training is seeded and replays on the same torch build and CPU (the
# heat run reproduces its committed loss history to every digit the CSV
# stores), but torch guarantees no bitwise determinism across versions or
# hardware -- which is why the artifacts are committed rather than regenerated.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY="python3"; fi
fi

echo "=================================================================="
echo "pinn-from-scratch: figure reproduction (no training)"
echo "python:     $("$PY" -V 2>&1)  ($PY)"
"$PY" - <<'EOF'
import importlib
for name in ("torch", "numpy", "matplotlib"):
    try:
        m = importlib.import_module(name)
        print(f"{name+':':11s} {m.__version__}")
    except ImportError:
        print(f"{name+':':11s} MISSING")
EOF
echo "=================================================================="

started=$SECONDS

step() {  # step <label> <script> [args...]
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $label"
    echo "------------------------------------------------------------------"
    local t0=$SECONDS
    "$PY" "$@"
    echo "    [${label}: $((SECONDS - t0))s]"
}

# The suite includes the checks that every figure has a replay path, that every
# artifact it reads is committed, and that every checkpoint still loads into the
# model the experiment builds today -- the failures this script exists to catch.
step "test suite" -m pytest -q
step "regenerate all 19 figures from committed artifacts" experiments/reproduce_figures.py

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="
