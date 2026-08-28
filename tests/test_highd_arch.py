"""Sec. 16: the architecture sweep, its design, and the claim it rests on.

Three kinds of test here. The design tests pin the one-factor-at-a-time
structure (a cell that changes two things at once would make the panels
uninterpretable, and nothing else would notice). The resume tests pin the
property the time-boxing depends on: a cell is a pure function of its key, so
an interrupted sweep continued later is the same sweep. And the last two assert
the *measured* result off the committed log, because Sec. 16's whole content is
a split -- the activation fixes the fit and not the generalization -- and a
refactor that quietly moved either half should fail here rather than in a
README nobody re-reads.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

pytest.importorskip("torch")

import highd_arch as ha  # noqa: E402
from common import read_csv  # noqa: E402
from highd_heat import HighDHeat  # noqa: E402


# --- design -----------------------------------------------------------------
def test_every_config_changes_exactly_one_thing_from_the_baseline():
    """One factor at a time is the design; two at a time is a different study."""
    for axis, width, depth, activation in ha.configs():
        differences = sum(x != y for x, y in
                          zip((width, depth, activation), ha.BASE))
        if axis == "base":
            assert differences == 0
        else:
            assert differences == 1, (axis, width, depth, activation)


def test_the_baseline_appears_exactly_once():
    """It belongs to all three axes; running it three times would waste a cell
    and, worse, invite three different numbers for one configuration."""
    base = [c for c in ha.configs() if tuple(c[1:]) == ha.BASE]
    assert len(base) == 1 and base[0][0] == "base"


def test_each_axis_covers_its_settings():
    got = {axis: {c[1:] for c in ha.configs() if c[0] == axis}
           for axis in ("width", "depth", "activation")}
    assert len(got["width"]) == len(ha.WIDTHS) - 1
    assert len(got["depth"]) == len(ha.DEPTHS) - 1
    assert len(got["activation"]) == len(ha.ACTIVATIONS) - 1


def test_axis_rows_reattaches_the_shared_baseline():
    """The baseline is logged under axis "base", so an axis that did not pull it
    back in would plot two points where it should plot three."""
    rows = read_csv(ha.ARCH_CSV)
    for axis, values in (("width", ha.WIDTHS), ("depth", ha.DEPTHS),
                         ("activation", ha.ACTIVATIONS)):
        for d in ha.DIMS:
            settings = [s for s, _ in ha.axis_rows(rows, axis, d)]
            assert set(settings) == set(values), (axis, d, settings)


# --- the committed log ------------------------------------------------------
def test_the_committed_log_has_every_cell_the_sweep_declares():
    rows = read_csv(ha.ARCH_CSV)
    have = {(int(r["d"]), int(r["width"]), int(r["depth"]), r["activation"],
             int(r["seed"])) for r in rows}
    want = set()
    for d in ha.DIMS:
        for _, w, dep, act in ha.configs():
            want |= {(d, w, dep, act, s) for s in ha.SEEDS}
    for d in ha.EXTRA_DIMS:
        for axis, w, dep, act in ha.configs():
            if axis in ("base", "activation"):
                want |= {(d, w, dep, act, s) for s in ha.SEEDS}
    assert want - have == set(), f"missing cells: {sorted(want - have)}"


def test_a_finished_sweep_reruns_nothing():
    """The resume key is what makes the time-boxed run safe to repeat."""
    before = read_csv(ha.ARCH_CSV)
    after = ha.sweep(write=False, verbose=False)
    assert len(after) == len(before)


def test_a_cell_is_a_pure_function_of_its_key():
    """What "resume is exact" means: rerunning a cell reproduces it.

    Cheap enough to actually run (d = 1, 20 steps), and it is the assumption
    every interrupted sweep on this host depends on.
    """
    problem = HighDHeat(1)
    a = ha.cell(problem, 32, 2, "tanh", seed=0, axis="width", steps=20)
    b = ha.cell(problem, 32, 2, "tanh", seed=0, axis="width", steps=20)
    assert a["rel_l2"] == b["rel_l2"]
    assert a["rel_fit_sampled"] == b["rel_fit_sampled"]


def test_seeds_actually_differ():
    """The guard against a seed argument that is accepted and ignored, which
    would report three copies of one number as a spread of zero."""
    problem = HighDHeat(1)
    a = ha.cell(problem, 32, 2, "tanh", seed=0, axis="width", steps=20)
    b = ha.cell(problem, 32, 2, "tanh", seed=1, axis="width", steps=20)
    assert a["rel_l2"] != b["rel_l2"]


# --- the result -------------------------------------------------------------
def _median(rows, d, activation, key):
    cells = [r for r in rows if int(r["d"]) == d and r["activation"] == activation
             and (int(r["width"]), int(r["depth"])) == ha.BASE[:2]]
    return float(np.median([float(r[key]) for r in cells]))


def test_at_d16_the_activation_fixes_the_fit_and_not_the_generalization():
    """Sec. 16's headline, and the reason it is not "sin is better".

    Sec. 14 reported that at d = 16 the regression control cannot fit even its
    own 4000 labels. With a sine activation it fits them to ~0.067 -- an order
    of magnitude better than tanh -- while its uniform-L2 test error stays
    within a factor of ~1.2 of tanh's and above 0.9, which is the region where
    a network that outputs zero scores 1.0. So the architecture was binding on
    the fit and is not binding on the generalization.
    """
    rows = read_csv(ha.ARCH_CSV)
    fit_sin = _median(rows, 16, "sin", "rel_fit_sampled")
    fit_tanh = _median(rows, 16, "tanh", "rel_fit_sampled")
    err_sin = _median(rows, 16, "sin", "rel_l2")
    err_tanh = _median(rows, 16, "tanh", "rel_l2")

    assert fit_tanh / fit_sin > 8, (fit_tanh, fit_sin)
    assert fit_sin < 0.15, fit_sin
    assert err_tanh / err_sin < 1.5, (err_tanh, err_sin)
    assert err_sin > 0.8, err_sin


def test_the_activation_advantage_does_not_grow_with_dimension():
    """The control that stops this from being read as "SIREN fixes high d".

    The target is a sum of products of sines, so a sine activation is matched
    to it, and a win could be a fact about the target rather than about
    dimension. Across d = 1..8 the test-error ratio stays inside a narrow band
    -- it is a roughly constant factor, not something that grows -- and by
    d = 16 it is nearly gone.
    """
    rows = read_csv(ha.ARCH_CSV)
    ratios = {d: _median(rows, d, "tanh", "rel_l2") / _median(rows, d, "sin", "rel_l2")
              for d in (1, 2, 4, 8)}
    assert all(2.5 < r < 9 for r in ratios.values()), ratios
    assert max(ratios.values()) / min(ratios.values()) < 3, ratios
    # And it does not survive to d = 16.
    r16 = _median(rows, 16, "tanh", "rel_l2") / _median(rows, 16, "sin", "rel_l2")
    assert r16 < min(ratios.values()), (r16, ratios)


def test_matched_params_is_not_vacuous():
    """It returns [] on this sweep, and the table says so. Check the function
    can find a pair at all, so "no pairs" means the design and not a bug."""
    rows = read_csv(ha.ARCH_CSV)
    assert ha.matched_params(rows) == []

    fake = [{"d": "4", "width": "512", "depth": "2", "activation": "tanh",
             "rel_l2": "1.0"},
            {"d": "4", "width": "128", "depth": "8", "activation": "tanh",
             "rel_l2": "2.0"}]
    found = ha.matched_params(fake, tol=10.0)
    assert len(found) == 1 and found[0]["d"] == 4
