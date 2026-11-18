"""
How strongly membership responds to price, estimated from past price rises.

A subscription service that has raised prices in some markets and not others holds a natural
experiment. This module estimates the price elasticity of membership from a monthly panel of
markets (market, month, price, members) in two ways:

    before-and-after   the change in members in markets that raised prices, from the six months
                       before the rise to months 6 to 11 after it, per unit change in log price.
                       Simple, but it attributes to the price rise anything else that moved
                       membership over the same months, such as growth common to all markets.

    event study        a stacked difference-in-differences comparison (Cengiz, Dube, Lindner and
                       Zipperer, 2019). For each month in which a price rise took place, the
                       markets that raised prices are compared with markets that did not change
                       price over the comparison window (never treated, or not yet treated).
                       Each comparison has its own market and month effects, so markets that
                       raised prices earlier never act as controls for later ones, which avoids
                       the bias of a single two-way fixed effects regression when rises are
                       staggered and responses build up over time (Goodman-Bacon, 2021).
                       Treatment is the size of the rise in log price, so each coefficient is an
                       elasticity at a given number of months after the rise. Estimates before
                       the rise test whether the two groups were on the same path beforehand.
                       Standard errors are clustered by market, since a comparison market can
                       appear in several comparisons.

The long-run elasticity is the average response 9 to 12 months after the rise, the point by which
cancellations have worked through. It is reported as a positive number: an elasticity of 1.2
means a 1% price rise loses 1.2% of members.

simulate_panel() writes the illustrative panel in examples/; its true long-run elasticity is 1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "examples" / "illustrative_price_panel.csv"
PANEL_COLUMNS = ["market", "month", "price", "members_k"]
PRE, POST = 6, 12            # months shown before and after each rise
LONG_RUN = range(9, 13)      # months averaged for the long-run response


# --------------------------------------------------------------------------------------
# Illustrative data
# --------------------------------------------------------------------------------------
def simulate_panel(seed: int = 20261118, n_markets: int = 30, n_never: int = 8, n_months: int = 48,
                   elasticity: float = 1.2, adjust_months: float = 2.0) -> pd.DataFrame:
    """Illustrative monthly panel. Membership grows in every market (common growth and seasonality,
    plus market-specific drift and persistent noise). Most markets raise their price once, by 6% to
    15%, at different months; the rest never change price. After a rise, membership falls by
    elasticity x the change in log price, building up over the following months as cancellations
    work through."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2022-01", periods=n_months, freq="M")
    t = np.arange(n_months)
    common = 0.008 * t + 0.012 * np.sin(2 * np.pi * (t - 2) / 12)
    rows = []
    for m in range(n_markets):
        treated = m >= n_never
        rise_at = int(rng.integers(PRE + 6, n_months - POST - 1)) if treated else None
        rise = float(rng.uniform(0.06, 0.15)) if treated else 0.0
        base_price = float(rng.choice([6.99, 7.99, 8.99, 9.99, 10.99, 11.99, 12.99]))
        level = float(rng.uniform(np.log(150), np.log(3000)))
        drift = rng.normal(0, 0.0008)
        noise = np.zeros(n_months)
        for i in range(n_months):
            noise[i] = (0.5 * noise[i - 1] if i else 0.0) + rng.normal(0, 0.005)
        for i in range(n_months):
            after = treated and i >= rise_at
            k = i - rise_at if after else -1
            response = -elasticity * rise * (1 - np.exp(-(k + 1) / adjust_months)) if after else 0.0
            price = round(base_price * np.exp(rise), 2) if after else base_price
            log_members = level + common[i] + drift * i + response + noise[i]
            rows.append({"market": f"Market {m + 1:02d}", "month": str(months[i]), "price": price,
                         "members_k": round(float(np.exp(log_members)), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------
def check_panel(panel: pd.DataFrame) -> list[str]:
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        return [f"Membership data is missing columns: {missing}. Expected {PANEL_COLUMNS}"]
    errors = []
    for col in ("price", "members_k"):
        v = pd.to_numeric(panel[col], errors="coerce")
        if (v.isna() | (v <= 0)).any():
            errors.append(f"{col} must be a positive number in every row")
    try:
        pd.PeriodIndex(panel["month"].astype(str), freq="M")
    except Exception:
        errors.append("month must be written as YYYY-MM")
    if errors:
        return errors
    if panel.duplicated(["market", "month"]).any():
        errors.append("Each market can appear only once in each month")
    counts = panel.groupby("market")["month"].nunique()
    if counts.nunique() > 1 or counts.iloc[0] != panel["month"].nunique():
        errors.append("Every market needs a row for every month (a balanced panel)")
    p = panel.sort_values(["market", "month"]).assign(lp=np.log(pd.to_numeric(panel["price"])))
    steps = p.groupby("market")["lp"].diff().fillna(0)
    if (steps < -1e-9).any():
        errors.append(f"Price falls found in {sorted(p.loc[steps < -1e-9, 'market'].unique())}; "
                      "the method uses price rises only")
    n_changes = (steps.abs() > 1e-9).groupby(p["market"]).sum()
    if (n_changes > 1).any():
        errors.append(f"Markets with more than one price change: {sorted(n_changes[n_changes > 1].index)}. "
                      "Split their data so each market-period has at most one rise")
    return errors


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    errors = check_panel(panel)
    if errors:
        raise ValueError("Membership data problems:\n- " + "\n- ".join(errors))
    p = panel.copy()
    p["period"] = pd.PeriodIndex(p["month"].astype(str), freq="M")
    p = p.sort_values(["market", "period"]).reset_index(drop=True)
    p["t"] = (p["period"] - p["period"].min()).apply(lambda x: x.n)
    p["lp"] = np.log(pd.to_numeric(p["price"]))
    p["y"] = np.log(pd.to_numeric(p["members_k"]))
    step = p.groupby("market")["lp"].diff().fillna(0)
    rises = p.loc[step > 1e-9, ["market", "t"]].assign(rise=step[step > 1e-9].values)
    p = p.merge(rises.rename(columns={"t": "rise_t"}), on="market", how="left")
    return p


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------
@dataclass
class ElasticityResult:
    long_run: float               # elasticity, positive number
    long_run_se: float
    before_after: float           # simple before-and-after elasticity
    path: pd.DataFrame            # months since rise, elasticity, se, low, high
    n_rising: int                 # markets whose rise is used
    n_comparison: int             # markets that never changed price in the data
    n_dropped: int                # rises too close to the start or end of the data
    n_obs: int

    @property
    def low(self) -> float:
        return self.long_run - 1.96 * self.long_run_se

    @property
    def high(self) -> float:
        return self.long_run + 1.96 * self.long_run_se


def before_after(p: pd.DataFrame, pre: int = PRE) -> float:
    num = den = 0.0
    for _, g in p[p["rise"].notna()].groupby("market"):
        r, rise = int(g["rise_t"].iloc[0]), float(g["rise"].iloc[0])
        before = g[(g["t"] >= r - pre) & (g["t"] < r)]["y"]
        after = g[(g["t"] >= r + 6) & (g["t"] < r + 12)]["y"]
        if len(before) == pre and len(after) == 6:
            num += after.mean() - before.mean()
            den += rise
    return -num / den if den else np.nan


def _demean_two_way(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Exact two-way demeaning for a balanced panel: x - mean by market - mean by month + mean."""
    out = df[cols] - df.groupby("market")[cols].transform("mean") - df.groupby("t")[cols].transform("mean")
    return out + df[cols].mean()


def event_study(panel: pd.DataFrame, pre: int = PRE, post: int = POST) -> ElasticityResult:
    p = prepare_panel(panel)
    t_max = int(p["t"].max())
    rise_t = p.groupby("market")["rise_t"].first()
    rise_size = p.groupby("market")["rise"].first()
    never = rise_t[rise_t.isna()].index.tolist()
    usable = rise_t[(rise_t >= pre) & (rise_t + post <= t_max)].dropna()
    ks = [k for k in range(-pre, post + 1) if k != -1]

    stacks = []
    for s, c in enumerate(sorted(usable.unique())):
        treated = usable[usable == c].index.tolist()
        later = rise_t[rise_t > c + post].index.tolist()          # not yet treated in the window
        controls = never + later
        if not controls:
            continue
        w = p[p["market"].isin(treated + controls) & p["t"].between(c - pre, c + post)].copy()
        w["k"] = np.where(w["market"].isin(treated), w["t"] - c, np.nan)
        w["d"] = np.where(w["market"].isin(treated), w["market"].map(rise_size), 0.0)
        for k in ks:
            w[f"x{k}"] = np.where(w["k"] == k, w["d"], 0.0)
        cols = ["y"] + [f"x{k}" for k in ks]
        w[cols] = _demean_two_way(w, cols)
        w["stack"] = s
        stacks.append(w)
    if not stacks or len(never) + len(rise_t) < 4:
        raise ValueError("Too few markets: the method needs markets that raised prices and markets that did not "
                         "over the same months")
    st = pd.concat(stacks, ignore_index=True)
    X = st[[f"x{k}" for k in ks]].to_numpy()
    y = st["y"].to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((len(ks), len(ks)))
    groups = st["market"].to_numpy()
    for g in np.unique(groups):
        sg = X[groups == g].T @ u[groups == g]
        meat += np.outer(sg, sg)
    n, k_par, n_g = len(y), len(ks), len(np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * n_g / (n_g - 1) * (n - 1) / (n - k_par)
    se = np.sqrt(np.diag(V))

    path = pd.DataFrame({"months_since_rise": ks, "elasticity": -beta, "se": se})
    path = pd.concat([path, pd.DataFrame({"months_since_rise": [-1], "elasticity": [0.0], "se": [0.0]})])
    path = path.sort_values("months_since_rise").reset_index(drop=True)
    path["low"] = path["elasticity"] - 1.96 * path["se"]
    path["high"] = path["elasticity"] + 1.96 * path["se"]

    a = np.array([-1.0 / len(LONG_RUN) if k in LONG_RUN else 0.0 for k in ks])
    long_run = float(a @ beta)
    long_run_se = float(np.sqrt(a @ V @ a))
    treated_all = rise_t.dropna()
    return ElasticityResult(long_run, long_run_se, before_after(p), path, len(usable), len(never),
                            len(treated_all) - len(usable), n)


def estimate(path: Path = PANEL) -> ElasticityResult:
    return event_study(pd.read_csv(path, comment="#"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        df = simulate_panel()
        header = ("# Illustrative only. Monthly members (thousands) and monthly price in 30 hypothetical markets,\n"
                  "# 2022 to 2025. 22 markets raise their price once; 8 never do. Generated by elasticity.simulate_panel()\n"
                  "# with a true long-run price elasticity of membership of 1.2.\n")
        PANEL.write_text(header + df.to_csv(index=False), encoding="utf-8")
        print(f"wrote {PANEL}")
    r = estimate()
    print(f"Long-run elasticity {r.long_run:.2f} (95% range {r.low:.2f} to {r.high:.2f}); "
          f"before-and-after {r.before_after:.2f}; {r.n_rising} rising markets, {r.n_comparison} never changed price, "
          f"{r.n_dropped} rises too close to the ends of the data")
    print(r.path.round(3).to_string(index=False))
