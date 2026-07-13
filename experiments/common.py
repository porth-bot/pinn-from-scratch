"""Shared plotting + logging setup for the experiment scripts.

Figures are written to ``figures/`` and numeric logs to ``logs/`` (both
committed), so every table and plot in the README can be regenerated from the
committed CSVs without re-running the (slow) PINN training.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

_HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(_HERE, "..", "figures")
LOGDIR = os.path.join(_HERE, "..", "logs")


def savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.abspath(os.path.join(FIGDIR, name))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def write_csv(name, fieldnames, rows):
    """Write ``rows`` (list of dicts) to ``logs/<name>`` with a header."""
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.abspath(os.path.join(LOGDIR, name))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")
    return path


def read_csv(name):
    """Read ``logs/<name>`` back into a list of dicts (strings)."""
    path = os.path.abspath(os.path.join(LOGDIR, name))
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
