"""The README's tables, recomputed from the committed logs.

``tests/test_reproduce_figures.py`` proves every figure can be rebuilt from
``logs/``, and ``reproduce.sh`` rebuilds them. Neither looks at a number
*typed into a table*, and this README has seventeen result sections carrying
several hundred of them, most transcribed by hand from a sweep that has since
been extended (Sec. 11's metric convention changed in Sec. 13, Sec. 15's
baseline arrived in Sec. 17, Sec. 14 rewrote what Sec. 11 and 12 had guessed).

diffusion-from-scratch got the same instrument on the day it was tagged and it
found seven drifted numbers; grokking-transformer's found three. So this is
the third repo in the series to be given it, and what it covers is stated
rather than implied:

  covered   Secs. 1, 11, 12, 13, 15, 16, 17 -- the width/collocation sweeps
            and every high-dimensional section, which is the work that has
            been rewritten most
  uncovered Secs. 2-10 and 14, whose logs are here but whose tables this
            module does not read yet

A test that silently covered less than its name suggests would be worse than
no test, which is why the list above is in the docstring rather than in a
commit message.

It found seven wrong on its first run, and the pattern is the same one the
other two repos showed: none is a typo, and most are a rounding that went up.

  Sec. 11  cost per step at d = 4: 65.8 ms for a measured 65.75
  Sec. 12  the d = 16 mesh solve: 30.7 s for 30.645
  Sec. 13  the HJB relative residual at d = 1: 0.051 for 0.0505
  Sec. 15  the d'Alembert reference's initial displacement, quoted as holding
           to 2e-16 where the log says 4.4e-16
  Sec. 16  the depth-8 cell at d = 16: 1.052 for a median of 1.0515, and the
           width effect at d = 8: 1.36x for 1.34x
  Sec. 17  the sine convergence order at t = 2: 4.01, which is the finest
           pair of grids, where the other five cells of that table are
           medians over the three refinements

The last two are the ones worth the module on their own, because neither is a
rounding: they are a statistic taken one way in one cell and another way in
the next, which is invisible to a reader and to a proofreader and shows up
only when something recomputes it. Both conventions are now stated in the
README beside the table.

Where a number depends on a convention -- median or mean over seeds, which
column of a sweep, which window of a trace -- the convention is written into
the test too, because that is the part a reader cannot recover from the
number. Two tables are not lookups at all: Sec. 11's and Sec. 13's normalized
losses divide each loss by the exact energy of what it matches, and that
energy is a closed form living in the producer, so the producer's own
``normalized_losses`` is called rather than reimplemented here. Everything
else is stdlib.
"""

import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
README = (ROOT / "README.md").read_text()


# -- reading the README ------------------------------------------------------

def section(number):
    """The text of numbered results section ``number``."""
    parts = re.split(r"^### ", README, flags=re.M)
    hits = [p for p in parts if p.startswith(f"{number}. ")]
    assert len(hits) == 1, f"section {number} matched {len(hits)} headings"
    return hits[0]


def plain(cell):
    """A table cell without markdown emphasis, ticks or LaTeX decoration."""
    cell = cell.replace("**", "").replace("`", "").replace("$", "").strip()
    cell = cell.replace("\\times10^{", "e").replace("\\times", "x")
    cell = cell.replace("{", "").replace("}", "").replace("\\", "")
    cell = cell.replace("^", "").replace("×", "x")
    return cell.replace("−", "-").replace("–", "-").strip()


class Table:
    """A markdown table's header cells and body rows."""

    def __init__(self, header, body):
        self.header, self.body = header, body

    def __getitem__(self, i):
        return self.body[i]

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        return iter(self.body)

    def row(self, label):
        hits = [r for r in self.body if r[0] == label]
        assert len(hits) == 1, f"{label!r} matched {len(hits)} rows"
        return hits[0]


def table(text, index=0):
    """The ``index``-th markdown table in ``text``, header kept.

    Most of these tables put the swept variable in the header -- the dimension,
    the width, the accuracy target -- so reading a row without checking which
    column it landed in would be reading half the table.
    """
    tables, block = [], []
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            block.append([plain(c) for c in stripped.strip("|").split("|")])
            continue
        rule = [i for i, cells in enumerate(block)
                if all(set(c) <= set("-: ") and c for c in cells)]
        if rule:
            cut = rule[0]
            tables.append(Table(block[cut - 1] if cut else [], block[cut + 1:]))
        block = []
    assert len(tables) > index, f"wanted table {index}, found {len(tables)}"
    return tables[index]


def rounds_to(value, text):
    """Does ``value`` print as the README's cell, at the cell's own precision?

    Comparing at the printed precision is what makes a failure informative:
    the test fires when the digits a reader sees would change, not when a
    float moves in its last bit.
    """
    text = plain(text).rstrip("x").replace(",", "").strip()
    if text.endswith("%"):
        value, text = value * 100, text.rstrip("%")
    quoted = float(text)
    if "." not in text and text.endswith("0"):
        # "1270" claims three significant figures, not four
        digits = len(text.rstrip("0")) or 1
        unit = 10.0 ** (len(text) - digits)
    elif re.search(r"e-?\+?\d", text):
        mantissa, _, exponent = text.partition("e")
        unit = 10.0 ** (int(exponent) - len(mantissa.partition(".")[2]))
    else:
        unit = 10.0 ** -len(text.partition(".")[2])
    # half a unit, with a hair of slack: a value that lands exactly on the
    # boundary (0.0245 shown to three decimals) prints either way depending on
    # the rounding convention, and which digit appears there is not a
    # measurement
    return abs(value - quoted) <= unit / 2 + 1e-12 * max(1.0, abs(quoted))


def check(value, text, what):
    assert rounds_to(value, text), f"{what}: README says {text!r}, logs say {value!r}"


def prose(text):
    """Section text with line breaks flattened, so a sentence can be asserted
    as the sentence it is rather than as the wrapping it happens to have."""
    return re.sub(r"\s+", " ", text)


# -- reading the logs --------------------------------------------------------

def rows(name):
    with open(LOGS / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def by(name, *keys):
    """Rows of a sweep grouped by the given key columns, values kept as text."""
    out = {}
    for r in rows(name):
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def column(group, key):
    return [float(r[key]) for r in group]


# -- Sec. 1: the heat equation, and the sweeps that support it ---------------

def test_the_collocation_and_width_sweeps():
    text = section(1)
    collocation = {int(r["n_interior"]): r for r in rows("heat_collocation")}
    body = table(text, 0)
    for label, r in zip(body.header[1:], sorted(collocation)):
        assert int(label.rstrip("k")) * 1000 == r, (label, r)
    for j, n in enumerate(sorted(collocation)):
        check(float(collocation[n]["rel_l2"]), body.row("rel L2")[j + 1],
              f"rel L2 at {n} interior points")

    widths = {int(r["width"]): r for r in rows("heat_width")}
    body = table(text, 1)
    assert [int(c) for c in body.header[1:]] == sorted(widths)
    for j, width in enumerate(sorted(widths)):
        check(float(widths[width]["rel_l2"]), body.row("rel L2")[j + 1],
              f"rel L2 at width {width}")


def test_the_tail_band_that_qualifies_the_width_sweep():
    """The section's own counter-evidence: the reported number is one iterate
    of an oscillation whose band is up to 22.8x wide, so the width ordering is
    not established by the row above it. The band is what carries that, and it
    is read from the same window (steps 2000-3000) for every cell."""
    text = section(1)
    tail = {int(r["width"]): r for r in rows("heat_tail")}
    body = table(text, 2)
    for j, width in enumerate(sorted(tail)):
        r = tail[width]
        check(float(r["final"]), body.row("reported (step 3000)")[j + 1],
              f"reported value at width {width}")
        cell = body.row("min-max over steps 2000-3000")[j + 1]
        low, high = [s.strip() for s in cell.split(" - ")]
        check(float(r["min"]), low, f"tail minimum at width {width}")
        check(float(r["max"]), high, f"tail maximum at width {width}")
        check(float(r["band"]), body.row("band")[j + 1], f"band at width {width}")
        assert int(r["tail_from"]) == 2000, "the window the band is taken over"
    # and the claim the band exists to support: neighbouring cells' bands
    # overlap, so the ordering the row above reports is not established
    widths = sorted(tail)
    for lo, hi in zip(widths, widths[1:]):
        assert float(tail[hi]["max"]) > float(tail[lo]["min"]), (lo, hi)
    assert max(float(r["band"]) for r in tail.values()) > 20


# -- Sec. 11: the PINN across dimensions -------------------------------------

def test_the_high_dimensional_sweep_table():
    text = section(11)
    sweep = by("highd_pinn_sweep", "d")
    cost = {int(r["d"]): r for r in rows("highd_pinn_cost")}
    body = table(text, 0)
    for row in body:
        d = int(row[0])
        group = sweep[(str(d),)]
        assert len(group) == 3, f"d={d}: {len(group)} seeds"
        errors = column(group, "rel_l2")
        assert row[1] == f"{int(group[0]['params']):,}", f"d={d}: parameters"
        check(statistics.mean(errors), row[2], f"d={d}: mean relative L2")
        check(statistics.stdev(errors), row[3], f"d={d}: seed sd")
        check(max(errors) / min(errors), row[4], f"d={d}: max/min")
        se = statistics.mean(float(r["stderr"]) / float(r["rel_l2"]) for r in group)
        check(se, row[5], f"d={d}: Monte Carlo error of the metric")
        check(float(cost[d]["ms_per_step"]), row[6], f"d={d}: ms per step")
        assert int(cost[d]["repeats"]) == 3, "the median is over three passes"


def test_the_headline_ratios_of_the_high_dimensional_sweep():
    """The section's two headline numbers, and the control that makes them
    readable: a network that outputs zero scores exactly 1.0 on this metric,
    so d=16's 1.041 is worse than saying nothing."""
    text = section(11)
    sweep = by("highd_pinn_sweep", "d")
    means = {int(d[0]): statistics.mean(column(g, "rel_l2"))
             for d, g in sweep.items()}
    cost = {int(r["d"]): float(r["ms_per_step"]) for r in rows("highd_pinn_cost")}
    check(means[16] / means[1], "1270", "the error's rise from d=1 to d=16")
    check(cost[16] / cost[1], "8.2", "the cost per step's rise")
    assert means[16] > 1.0 > means[8], means
    check(1 - means[8], "0.24", "how much better than zero d=8 is")
    assert "*worse than saying nothing*" in prose(text)

    spreads = [max(column(g, "rel_l2")) / min(column(g, "rel_l2"))
               for g in sweep.values()]
    check(min(spreads), "1.03", "the smallest seed spread")
    check(max(spreads), "1.68", "the largest seed spread")
    assert "1.03–1.68×" in text


def experiment(name):
    """A results module, imported the way the other test modules import them.

    Two of these tables are not lookups: the normalized-loss rows divide each
    loss by the exact energy of the thing it matches, and that energy is a
    closed form living in the producer. Recomputing it here would be a second
    implementation that can drift on its own, so the producer's own
    ``normalized_losses`` is what the README is checked against.
    """
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    return __import__(name)


def test_the_losses_are_normalized_by_what_they_match():
    """Sec. 11's second table, and the reason it exists: the target's rms
    falls like 2^{-d/2}, so d=16 posts the second-smallest raw loss_ic in the
    sweep while fitting the initial condition worse than zero would."""
    text = section(11)
    sweep = by("highd_pinn_sweep", "d")
    normalized = {r["d"]: r
                  for r in experiment("highd_pinn").normalized_losses(
                      rows("highd_pinn_sweep"))}
    body = table(text, 1)
    dims = [int(c) for c in body.header[1:]]
    for j, d in enumerate(dims):
        r = normalized[d]
        check(float(r["rel_ic_error"]), body.row("relative IC error")[j + 1],
              f"d={d}: normalized IC error")
        check(float(r["rel_residual"]), body.row("relative residual")[j + 1],
              f"d={d}: normalized residual")
    assert rounds_to(float(sweep[("16",)][0]["exact_rms"]), "0.00346")
    assert rounds_to(float(sweep[("1",)][0]["exact_rms"]), "0.591")
    raw_ic = {int(d[0]): statistics.mean(column(g, "loss_ic"))
              for d, g in sweep.items()}
    assert sorted(raw_ic, key=raw_ic.get)[1] == 16, \
        "d=16 is meant to hold the second-smallest raw loss_ic"
    assert "second-smallest number in the whole sweep" in prose(text)


def test_selecting_the_iterate_by_training_loss_is_worth_what_it_says():
    """No ground truth enters the choice -- the selection is on training loss
    -- so the gain is free, and it is quoted per dimension."""
    text = prose(section(11))
    sweep = by("highd_pinn_sweep", "d")
    for d, quoted in ((1, "2.5"), (2, "3.1"), (4, "1.9"), (8, "1.01")):
        group = sweep[(str(d),)]
        ratio = (statistics.mean(column(group, "rel_l2_final"))
                 / statistics.mean(column(group, "rel_l2")))
        check(ratio, quoted, f"d={d}: final iterate against the selected one")
        assert f"{quoted}×" in text


# -- Sec. 12: the crossover ---------------------------------------------------

def test_the_crossover_headline_row():
    text = section(12)
    mesh = {int(r["d"]): r for r in rows("highd_crossover_mesh")
            if float(r["target"]) == 0.1}
    pinn = {int(r["d"]): r for r in rows("highd_crossover_pinn")
            if float(r["target"]) == 0.1}
    body = table(text, 0)
    dims = [int(c) for c in body.header[1:]]
    for j, d in enumerate(dims):
        check(float(mesh[d]["seconds"]), body.row("mesh, s")[j + 1],
              f"d={d}: mesh seconds")
        check(float(mesh[d]["rel_l2"]), body.row("mesh rel L2")[j + 1],
              f"d={d}: mesh relative L2")
    assert max(dims) == 16 and "measured" in prose(text)


def test_the_crossover_summary_table_is_a_lower_bound():
    """Every row is a bound in the same direction: the PINN is granted a step
    count that does not grow with d, so the dimension at which the mesh
    becomes dearer can only move up."""
    text = section(12)
    summary = {r["target"]: r for r in rows("highd_crossover_summary")}
    body = table(text, 1)
    assert len(body) == 3
    for row, target in zip(body, ("1e-01", "1e-02", "1e-03")):
        r = summary[target]
        assert row[1] == f"d \\geq {r['d_crossover_lower_bound']}".replace("\\", ""), \
            (target, row[1])
        assert row[2].endswith(r["d_pinn_last_reached"]), (target, row[2])
        assert row[3].split()[0] == r["headroom"], (target, row[3])
        assert int(r["headroom"]) == (int(r["d_crossover_lower_bound"])
                                      - int(r["d_pinn_last_reached"]))
    assert "lower bound" in prose(text)


# -- Sec. 13: a second PDE ----------------------------------------------------

def test_the_metric_error_table_for_both_problems():
    """The estimator's own precision, and the point of the table: it degrades
    with d on the heat problem (8.7% at d=16) and *improves* on the HJB one."""
    text = section(13)
    body = table(text, 0)
    dims = [int(c.lstrip("d=")) for c in body.header[1:]]
    for name, prefix in (("highd_metric", "heat"), ("highd_hjb_metric", "HJB")):
        logged = {int(r["d"]): r for r in rows(name) if int(r["n"]) == 100000}
        for j, d in enumerate(dims):
            check(float(logged[d]["pred_rel_sd"]),
                  body.row(f"{prefix}: rel. s.e. of the metric")[j + 1],
                  f"{prefix}: metric s.e. at d={d}")
            check(float(logged[d]["top1pct_share"]),
                  body.row(f"{prefix}: top 1% of points carry")[j + 1],
                  f"{prefix}: top-1% share at d={d}")
    heat = {int(r["d"]): float(r["pred_rel_sd"]) for r in rows("highd_metric")
            if int(r["n"]) == 100000}
    hjb = {int(r["d"]): float(r["pred_rel_sd"]) for r in rows("highd_hjb_metric")
           if int(r["n"]) == 100000}
    assert heat[16] > heat[1] and hjb[16] < hjb[1], (heat, hjb)


def test_the_hjb_sweep_table_and_its_trivial_predictors():
    """Three seeds cannot tell a tail from a regime, so the mean and the
    median are both reported -- the mean is non-monotone only because d=4 has
    an 8.5x seed spread."""
    text = section(13)
    sweep = by("highd_hjb_sweep", "d")
    body = table(text, 1)
    dims = [int(c) for c in body.header[1:]]
    for j, d in enumerate(dims):
        group = sweep[(str(d),)]
        assert len(group) == 3, f"d={d}: {len(group)} seeds"
        errors = column(group, "rel_sd")
        check(statistics.mean(errors), body.row("HJB, mean rel. error")[j + 1],
              f"d={d}: HJB mean")
        check(statistics.median(errors), body.row("median (3 seeds)")[j + 1],
              f"d={d}: HJB median")
        check(max(errors) / min(errors), body.row("seed spread (max/min)")[j + 1],
              f"d={d}: HJB seed spread")
        check(statistics.mean(column(group, "base_profile")),
              body.row("exact E_x[u mid t] scores")[j + 1],
              f"d={d}: the profile-only predictor")
        check(statistics.mean(column(group, "base_zero")),
              body.row("u=0 scores")[j + 1], f"d={d}: outputting zero")
        assert body.row("best constant scores")[j + 1] == "1.000"

    means = [statistics.mean(column(sweep[(str(d),)], "rel_sd")) for d in dims]
    medians = [statistics.median(column(sweep[(str(d),)], "rel_sd")) for d in dims]
    assert means != sorted(means), "the mean is non-monotone, which is the point"
    assert medians == sorted(medians), "and the median is not"
    assert "monotone" in prose(text)


def test_the_hjb_losses_are_normalized_too():
    text = section(13)
    sweep = by("highd_hjb_sweep", "d")
    normalized = {r["d"]: r
                  for r in experiment("highd_hjb").normalized_losses(
                      rows("highd_hjb_sweep"))}
    body = table(text, 2)
    dims = [int(c) for c in body.header[1:]]
    for j, d in enumerate(dims):
        group = sweep[(str(d),)]
        sd = statistics.mean(column(group, "exact_sd"))
        del group, sd
        r = normalized[d]
        for label, key in (("relative residual", "rel_residual"),
                           ("relative terminal", "rel_terminal"),
                           ("relative boundary", "rel_boundary")):
            check(float(r[key]), body.row(label)[j + 1], f"d={d}: {label}")


def test_the_hjb_ground_truth_is_checked_against_independent_computations():
    """The section rests on a closed-form value function, so the check table
    is what licenses everything else: the PDE residual, the Cole-Hopf
    identity, and quadrature against Monte Carlo all agree to machine
    precision or to the sampling error."""
    check_rows = {int(r["d"]): r for r in rows("highd_hjb_check")}
    for d, r in check_rows.items():
        assert float(r["pde_residual"]) < 1e-13, (d, r["pde_residual"])
        assert float(r["cole_hopf_rel"]) < 1e-13, (d, r["cole_hopf_rel"])
        assert float(r["quad_64_vs_256"]) < 1e-12, (d, r["quad_64_vs_256"])
        assert float(r["mc_mean_rel"]) < 1e-2, (d, r["mc_mean_rel"])
    assert "closed form" in prose(section(13))


# -- Sec. 15: the wave equation ----------------------------------------------

def test_the_wave_table_and_the_energy_deficit():
    """The conserved quantity is the diagnostic here, and the loss never
    references it -- which is what makes the 0.766 an independent read-out
    rather than a restatement of the error."""
    text = section(15)
    cells = by("wave_cells", "ic")
    body = table(text, 0)
    for row in body:
        group = cells[(row[0],)]
        assert len(group) == 3, f"{row[0]}: {len(group)} seeds"
        errors = column(group, "rel_l2")
        check(statistics.mean(errors), row[1], f"{row[0]}: mean relative L2")
        check(max(errors) / min(errors), row[2], f"{row[0]}: seed spread")
        check(statistics.mean(column(group, "energy_ratio")), row[3],
              f"{row[0]}: energy carried")
        check(statistics.mean(column(group, "kink_error")), row[4],
              f"{row[0]}: error near a corner")
        check(statistics.mean(column(group, "smooth_error")), row[5],
              f"{row[0]}: error elsewhere")

    pluck = statistics.mean(column(cells[("pluck",)], "rel_l2"))
    sine = statistics.mean(column(cells[("sine",)], "rel_l2"))
    check(pluck / sine, "8", "what the corner costs against the smooth control")
    concentration = (statistics.mean(column(cells[("pluck",)], "kink_error"))
                     / statistics.mean(column(cells[("pluck",)], "smooth_error")))
    check(concentration, "2.6", "how concentrated the error is on the corner")


def test_the_dalembert_reference_is_exact_at_the_precision_it_claims():
    """The ground truth is the odd 2-periodic extension, closed form for any
    initial displacement -- so it does not converge in the number of modes the
    way a sine series would, and the check asserts that rather than a
    tolerance."""
    text = prose(section(15))
    for r in rows("wave_check"):
        assert float(r["ic_error"]) < 5e-16, r
        assert float(r["initial_velocity"]) == 0.0, r
        assert max(float(r["bc_left"]), float(r["bc_right"])) < 5e-16, r
        assert float(r["residual_fd"]) < 9e-8, r
    # the sine series converges on a smooth f and Gibbs at a corner does not:
    # the rms gap falls with the mode count while the max gap barely moves,
    # which is the section's reason for not using the series as the reference
    pluck = {int(r["n_modes"]): r for r in rows("wave_check") if r["ic"] == "pluck"}
    modes = sorted(pluck)
    rms = [float(pluck[m]["rms_gap"]) for m in modes]
    top = [float(pluck[m]["max_gap"]) for m in modes]
    assert rms == sorted(rms, reverse=True) and top == sorted(top, reverse=True)
    assert rms[0] / rms[-1] > 30 > top[0] / top[-1], (rms, top)
    for value in (rms[0], rms[1], rms[2], top[0], top[-1]):
        assert f"{value:.1e}".replace("e-0", "\\times10^{-") + "}" in \
            section(15).replace("$", ""), value
    for r in rows("wave_check"):
        if r["ic"] == "sine":
            assert float(r["max_gap"]) < 1e-15, r
    assert "used here to *check* d'Alembert and never as the reference" in text


# -- Sec. 16: does a different network fix it --------------------------------

def test_the_architecture_grid():
    """Size does nothing worth having, and the grid is read against the seed
    spread of its own base cell rather than against zero."""
    text = section(16)
    logged = by("highd_arch", "axis", "d", "width", "depth", "activation")

    def median_error(axis, d, width, depth, activation="tanh"):
        # the grid's centre cell is logged once, under axis "base", rather
        # than twice under "width" and "depth"
        if width == 128 and depth == 4 and activation == "tanh":
            axis = "base"
        key = (axis, str(d), str(width), str(depth), activation)
        return statistics.median(column(logged[key], "rel_l2"))

    body = table(text, 0)
    for row in body:
        d = int(row[0])
        for j, label in enumerate(body.header[1:]):
            axis, value = label.split()
            kw = {"axis": axis, "d": d, "width": 128, "depth": 4}
            kw[axis] = int(value)
            check(median_error(**kw), row[j + 1], f"d={d}, {label}")

    # the spread the section compares its effects against is the widest cell's,
    # which is the depth-8 arm at d = 8 and not the base cell (1.05)
    spreads = {k: max(column(g, "rel_l2")) / min(column(g, "rel_l2"))
               for k, g in logged.items() if k[1] == "8"}
    widest = max(spreads, key=spreads.get)
    assert widest[3] == "8", f"the widest d=8 cell is {widest}"
    spread = spreads[widest]
    check(spread, "1.52", "the widest d=8 cell's seed spread")
    assert "reaches 1.52× in the depth-8 cell" in prose(text)
    widths = [median_error(axis="width", d=8, width=w, depth=4)
              for w in (32, 128, 512)]
    check(max(widths) / min(widths), "1.34", "what 16x the width buys at d=8")
    depths = [median_error(axis="depth", d=8, width=128, depth=p)
              for p in (2, 4, 8)]
    check(max(depths) / min(depths), "1.10", "what 4x the depth buys at d=8")
    assert max(max(widths) / min(widths), max(depths) / min(depths)) < spread


def test_the_activation_table_and_the_control_that_reads_it():
    """A sine network is *matched* to a target that is a sum of products of
    sines, so the axis the claim is about has to be swept: a constant factor
    across d is a representation match, a trend would be a high-dimensional
    fix. It is the former, and it is swamped by d=16."""
    text = section(16)
    logged = by("highd_arch", "axis", "d", "activation")
    body = table(text, 1)
    dims = [int(c) for c in body.header[1:]]
    for j, d in enumerate(dims):
        for label, activation, key in (
                ("tanh, test", "tanh", "rel_l2"),
                ("sin, test", "sin", "rel_l2"),
                ("tanh, own sample", "tanh", "rel_fit_sampled"),
                ("sin, own sample", "sin", "rel_fit_sampled")):
            group = (logged.get(("activation", str(d), activation))
                     or logged[("base", str(d), activation)])
            check(statistics.median(column(group, key)),
                  body.row(label)[j + 1], f"d={d}: {label}")

    def median(d, activation, key="rel_l2"):
        group = (logged.get(("activation", str(d), activation))
                 or logged[("base", str(d), activation)])
        return statistics.median(column(group, key))

    ratios = [median(d, "tanh") / median(d, "sin") for d in dims]
    for ratio, quoted in zip(ratios, ("7.4", "3.5", "5.3", "3.5", "1.19")):
        check(ratio, quoted, f"tanh/sin ratio")
    assert max(ratios[:-1]) / min(ratios[:-1]) < 2.2, "a constant factor, no trend"
    assert ratios[-1] < 1.5, "and it is swamped at d=16"
    check(median(16, "sin") / median(1, "sin"), "474", "sin's own rise over the range")
    check(median(16, "tanh") / median(1, "tanh"), "77", "tanh's")
    check(median(16, "sin") / median(16, "sin", "rel_fit_sampled"), "14.3",
          "the gap between what sin fits and what it generalizes at d=16")
    assert "representation match that exists at every dimension" in prose(text)


# -- Sec. 17: the leapfrog baseline -------------------------------------------

def test_the_convergence_order_depends_on_when_it_is_measured():
    """At Courant number 1 the scheme is exact on the characteristics, so the
    head-to-head is run at r < 1 on purpose; and the sine initial condition
    sits at a turning point at t = 1 and t = 2, where a phase error enters
    quadratically and a second-order scheme reads 4."""
    text = section(17)
    orders = {}
    for r in rows("wave_leapfrog_order"):
        if r["order"]:
            orders.setdefault((r["ic"], float(r["t_end"])), []).append(float(r["order"]))
    body = table(text, 0)
    for row in body:
        for j, label in enumerate(body.header[1:]):
            t = float(label.split("=")[1])
            # the median of the three successive-refinement estimates, because
            # the finest pair alone moves the sine cells by 0.03 and the pluck
            # cells by 0.02 and the table is quoting a converged order
            measured = statistics.median(orders[(row[0], t)])
            check(measured, row[j + 1], f"{row[0]} at t={t}")
    assert "median of the three refinements" in prose(text)

    sine = statistics.median(orders[("sine", 0.7)])
    turning = statistics.median(orders[("sine", 2.0)])
    assert sine < 2.5 < turning, (sine, turning)
    assert statistics.median(orders[("pluck", 0.7)]) < 1.5, \
        "the corner costs the scheme an order"

    exact = [r for r in rows("wave_leapfrog") if float(r["cfl"]) == 1.0]
    assert exact and all(float(r["rel_l2"]) < 1e-14 for r in exact), \
        "at r = 1 the update is d'Alembert on the characteristics"
    assert "**At $r = 1$ it is exact.**" in prose(text)


def test_the_head_to_head_against_the_network():
    """Sec. 15 predicted the mesh would win on every axis and it does; the
    comparison is at r < 1, and the network's seconds are its own training
    time from Sec. 15's cells."""
    text = section(17)
    cells = by("wave_cells", "ic")
    mesh = {r["ic"]: r for r in rows("wave_leapfrog")
            if float(r["cfl"]) == 0.5 and int(r["nx"]) == 26}
    assert "$r = 0.5$ and the *coarsest grid in the sweep* ($n_x = 26$)" in text
    body = table(text, 1)
    for row in body:
        ic = row[0]
        group = cells[(ic,)]
        pinn_error = statistics.mean(column(group, "rel_l2"))
        pinn_seconds = statistics.mean(column(group, "train_seconds"))
        best = mesh[ic]
        check(pinn_error, row[1], f"{ic}: PINN relative L2")
        check(pinn_seconds, row[2], f"{ic}: PINN seconds")
        check(float(best["rel_l2"]), row[3], f"{ic}: mesh relative L2")
        check(float(best["seconds"]), row[4], f"{ic}: mesh seconds")
        check(pinn_error / float(best["rel_l2"]), row[5], f"{ic}: accuracy ratio")
        check(pinn_seconds / float(best["seconds"]), row[6], f"{ic}: speed ratio")
