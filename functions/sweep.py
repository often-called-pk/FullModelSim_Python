"""Pure (MPI-free, casadi-free) helpers for the MLTP sweep driver.

run_sweep.py is a thin MPI shell over these functions; everything that needs a
decision or formatting lives here so it can be unit-tested without MPI or
CasADi. The only injected dependency is the per-case solve function (the real
MLTP in production, a stub in tests).
"""
import csv
import os

# Columns a user may set in cases.csv: explicit MLTP() params they may vary,
# plus userOpts passthroughs (linear_solver reaches MLTP via **useropts_kwargs).
ACCEPTED_COLUMNS = (
    "circuit", "vi", "ni", "warm_start", "AeroConfig", "ATD",
    "Electric_4Motors", "TyreModel", "linear_solver",
)
_FLOAT_COLUMNS = ("vi", "ni")


def read_cases(path):
    """Parse cases.csv into a list of case dicts.

    Blank cells are dropped (so the MLTP/userOpts default applies). Each case
    gets a string 'case_id': the 'case_id' column if present, else the
    zero-based row index.
    """
    cases = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            case = {k: v.strip() for k, v in row.items()
                    if k is not None and v is not None and v.strip() != ""}
            case["case_id"] = case.get("case_id", str(i))
            cases.append(case)
    return cases


def case_kwargs(case):
    """Map a case dict to MLTP() kwargs.

    Validates columns against ACCEPTED_COLUMNS, coerces float columns, drops
    case_id, and forces plot=False. Raises ValueError on an unknown column.
    """
    unknown = set(case) - set(ACCEPTED_COLUMNS) - {"case_id"}
    if unknown:
        raise ValueError(
            f"case {case.get('case_id', '?')}: unknown column(s) "
            f"{sorted(unknown)}; accepted: {sorted(ACCEPTED_COLUMNS)}")
    kwargs = {}
    for k, v in case.items():
        if k == "case_id":
            continue
        kwargs[k] = float(v) if k in _FLOAT_COLUMNS else v
    kwargs["plot"] = False
    return kwargs


# Manifest outcome columns, in order, appended after the input columns.
OUTCOME_FIELDS = ("status", "return_status", "iter_count",
                  "lap_time", "init_s", "solve_s", "wall_s", "out_path")


def output_dir_for(sweep_name, case_id):
    """Per-case results dir: Results/<sweep_name>/case_<case_id>/.

    Unique per case even when two cases differ only by vi (which MLTP's own
    <circuit>_<config>.mat filename omits).
    """
    return os.path.join("Results", sweep_name, f"case_{case_id}")


def input_fieldnames(cases):
    """Ordered input columns across all cases: case_id first, then
    ACCEPTED_COLUMNS in declared order, keeping only those that appear."""
    present = set()
    for c in cases:
        present.update(c)
    return ["case_id"] + [c for c in ACCEPTED_COLUMNS if c in present]


def manifest_fieldnames(cases):
    """Full manifest header: input columns then outcome columns."""
    return input_fieldnames(cases) + list(OUTCOME_FIELDS)


def manifest_row(case, result):
    """Flatten a case's inputs and its outcome into one manifest row dict."""
    row = dict(case)
    row.update(result)
    return row
