"""
UK economic footprint model: a static input-output model of the UK economy built from the
ONS Input-Output Analytical Tables (IOATs) for 2023.

It estimates the output, gross value added (GVA), compensation of employees and full-time
equivalent (FTE) employment supported by a pattern of spending in the UK, split into:

    direct    activity carried out by the organisation itself
    indirect  activity in its UK supply chain (Type I)
    induced   activity supported when the wages earned in the first two are spent (Type II)

The Type I engine reproduces the ONS published Leontief inverse, output multipliers, GVA
effects and FTE effects for all 104 industries; run `python io_footprint.py validate`.

Price basis. Spending inputs are at purchasers' prices, net of recoverable VAT, in the prices of the
year in which the money was spent. The bridge to basic prices removes distribution margins (which
are re-routed to wholesale) and the non-deductible product taxes the purchasing industry pays on its
inputs, at that industry's own ONS rate. Results are in the same prices as the inputs. FTE
coefficients are 2023 values, so set price_index to the price level of the input year relative to
2023 when inputs are not in 2023 prices.

Author: Gideon Morrison-Wood. Code released under the MIT licence. Source data are
Crown copyright (ONS) and reproduced under the Open Government Licence v3.0; see README.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
NO_LEONTIEF = ["L68A"]  # owner-occupiers' housing: imputed rent, no ONS Leontief column


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclass
class Settings:
    """Model settings. Defaults are the central case, with sources noted."""

    # Household closure for induced (Type II) effects
    tax_wedge: float = 0.312      # OECD Taxing Wages 2025, UK average single worker, 2023
    saving_ratio: float = 0.062   # ONS Blue Book 2025, household saving ratio, 2023
    include_induced: bool = True

    # Purchasers' price to basic price bridge
    remove_margins: bool = True        # A10 distributors' trading margin rates (Blue Book 2025)
    remove_product_taxes: bool = True  # non-deductible product taxes paid on purchases
    purchaser_industry: str | None = None  # IOAT code whose ONS product-tax rate on intermediate
                                           # purchases applies; None uses all industries
    margin_destination: dict = field(default_factory=lambda: {"G46": 1.0})

    # Import leakage
    remove_imports: bool = True
    import_uplift: float = 0.0         # proportional uplift to import shares (sensitivity)

    # Direct effect of in-house staff costs (spend_type in_house_pay): operating surplus added
    # to pay as a share of pay. Default 0, so direct GVA equals pay (a lower bound).
    direct_margin_on_pay: float = 0.0

    # Price level of the input year relative to 2023. FTE coefficients are 2023 values, so FTE
    # are divided by this index; monetary results stay in the prices of the inputs.
    price_index: float = 1.0

    @property
    def kappa(self) -> float:
        """Household consumption per £ of compensation of employees."""
        return (1 - self.tax_wedge) * (1 - self.saving_ratio)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


@dataclass
class ONSData:
    codes: list
    names: pd.Series
    a10: pd.Series
    Z: pd.DataFrame          # domestic intermediate use, basic prices, £m (industry x industry)
    output: pd.Series        # P1, £m
    gva: pd.Series
    coe: pd.Series           # D1 compensation of employees
    gos: pd.Series           # gross operating surplus and mixed income
    ptax: pd.Series          # taxes less subsidies on production (D29-D39)
    hh_product_taxes: float  # taxes less subsidies on products paid on household consumption, £m
    int_product_taxes: pd.Series  # taxes less subsidies on products paid on intermediate purchases
    inputs_pp: pd.Series     # total intermediate use at purchasers' prices
    total_use: pd.Series     # row total use (should equal output)
    hhfce_dom_bp: pd.Series  # household final consumption of UK output, basic prices
    fte_per_m: pd.Series
    ons_leontief: pd.DataFrame
    ons_effects: pd.DataFrame
    ons_fte: pd.DataFrame
    imports: pd.DataFrame
    sut_a10: pd.DataFrame
    bluebook: dict
    hh_params: dict


def load_data(data_dir: Path = DATA_DIR) -> ONSData:
    ind = pd.read_csv(data_dir / "industries.csv")
    codes = ind["code"].tolist()

    iot = pd.read_csv(data_dir / "ons_iot_2023_domestic_use.csv", dtype={"row_code": str})
    iot["row_code"] = iot["row_code"].fillna("")
    body = iot.iloc[:104].set_index("row_code")
    assert body.index.tolist() == codes, "IOT row order does not match industries.csv"
    Z = _num(body[codes])

    def row(code=None, label=None):
        if code is not None:
            r = iot[iot["row_code"] == code]
        else:
            r = iot[iot["row_label"] == label]
        return _num(r[codes]).iloc[0]

    eff = pd.read_csv(data_dir / "ons_type1_effects_2023.csv").set_index("code")
    fte = pd.read_csv(data_dir / "ons_fte_2023.csv").set_index("code")
    leon = pd.read_csv(data_dir / "ons_leontief_type1_2023.csv").set_index("code")
    imp = pd.read_csv(data_dir / "ons_imports_pxp_2023.csv").set_index("code")
    sut = pd.read_csv(data_dir / "ons_sut_a10_2023.csv").set_index("a10_group")
    bb = pd.read_csv(data_dir / "ons_bluebook_2023_totals.csv")
    hh = pd.read_csv(data_dir / "hhfce_closure_2023.csv")

    for t in (eff, fte, leon, imp):
        assert t.index.tolist() == codes, "code misalignment in ONS tables"

    return ONSData(
        codes=codes,
        names=ind.set_index("code")["name"],
        a10=ind.set_index("code")["a10_group"],
        Z=Z,
        output=row("P1"),
        gva=row("GVA"),
        coe=row("D1"),
        gos=row("GOS & MI"),
        ptax=row("D29-D39"),
        hh_product_taxes=float(_num(iot.loc[iot["row_label"] == "Taxes less subsidies on products", ["P3 S14"]]).iloc[0, 0]),
        int_product_taxes=row(label="Taxes less subsidies on products"),
        inputs_pp=row(label="Total intermediate use at purchaser's prices"),
        total_use=_num(body[["TU"]])["TU"],
        hhfce_dom_bp=_num(body[["P3 S14"]])["P3 S14"],
        fte_per_m=_num(fte[["fte_per_gbp_m"]])["fte_per_gbp_m"],
        ons_leontief=_num(leon),
        ons_effects=_num(eff),
        ons_fte=_num(fte),
        imports=_num(imp.drop(columns=["cpa_code"])),
        sut_a10=_num(sut),
        bluebook=dict(zip(bb["item"], bb["value_gbp_m"])),
        hh_params=dict(zip(hh["item"], hh["value_gbp_m"])),
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class FootprintModel:
    def __init__(self, settings: Settings | None = None, data: ONSData | None = None):
        self.s = settings or Settings()
        self.d = data or load_data()
        d, n = self.d, len(self.d.codes)
        self.n = n
        x = d.output.values

        # Technical coefficients and Type I Leontief inverse
        with np.errstate(divide="ignore", invalid="ignore"):
            self.A = np.where(x > 0, d.Z.values / x, 0.0)
            self.v = np.where(x > 0, d.gva.values / x, 0.0)   # GVA per £ output
            self.w = np.where(x > 0, d.coe.values / x, 0.0)   # CoE per £ output
            self.g = np.where(x > 0, d.gos.values / x, 0.0)   # operating surplus per £ output
            self.t_prod = np.where(x > 0, d.ptax.values / x, 0.0)  # production taxes per £ output
        self.e = d.fte_per_m.values                            # FTE per £m output
        self.L1 = np.linalg.inv(np.eye(n) - self.A)
        self.in_leontief = np.array([c not in NO_LEONTIEF for c in d.codes], dtype=float)

        # Household closure: household consumption pattern s(i), basic prices, UK output,
        # as a share of household spending at purchasers' prices, both excluding imputed rent
        hp = d.hh_params
        hh_pp_ex_rent = (
            hp["Total HHFCE at purchasers' prices (£m)"]
            - hp["Imputed rental: domestic output consumed by households (£m)"]
            - hp["Imputed rental: imported, consumed by households (£m)"]
            - hp["Taxes less subsidies on imputed rental (£m)"]
        )
        self.s_hh = d.hhfce_dom_bp.values * self.in_leontief / hh_pp_ex_rent
        # product taxes (VAT, duties) per £ of household spending, excluding imputed rent
        self.hh_product_tax_rate = (d.hh_product_taxes - hp["Taxes less subsidies on imputed rental (£m)"]) / hh_pp_ex_rent
        self.c = self.s.kappa * self.s_hh

        A_star = np.zeros((n + 1, n + 1))
        A_star[:n, :n] = self.A
        A_star[:n, n] = self.c
        A_star[n, :n] = self.w
        self.L2 = np.linalg.inv(np.eye(n + 1) - A_star)

        # Price and import bridge parameters
        sut = d.sut_a10
        margin_rate = (sut["trade_margins"].clip(lower=0) / sut["total_supply_pp"]).fillna(0)
        self.margin_rate = d.a10.map(margin_rate).fillna(0).values
        # Non-deductible product taxes per £ of purchases at purchasers' prices, measured for the
        # purchasing industry from the same ONS table as the supply chain (film and TV: 0.14%)
        code = self.s.purchaser_industry
        if code is None:
            rate = d.int_product_taxes.sum() / d.inputs_pp.sum()
        elif code in d.codes:
            rate = d.int_product_taxes[code] / d.inputs_pp[code]
        else:
            raise KeyError(f"Unknown purchaser_industry: {code}")
        self.purchase_tax_rate = max(0.0, float(rate))
        im = d.imports
        with np.errstate(divide="ignore", invalid="ignore"):
            tot_int = im["domestic_intermediate"] + im["imported_intermediate"]
            tot_all = im["domestic_total_use"] + im["imported_total_use"]
            self.mu_proc = np.where(tot_int > 0, im["imported_intermediate"] / tot_int, 0.0)
            self.mu_capex = np.where(tot_all > 0, im["imported_total_use"] / tot_all, 0.0)

    # ---- helpers -------------------------------------------------------------------
    def vec(self, values: dict | pd.Series | None) -> np.ndarray:
        """Convert {IOAT code: £m} to a vector in ONS industry order."""
        out = np.zeros(self.n)
        if values is None:
            return out
        idx = {c: i for i, c in enumerate(self.d.codes)}
        for k, val in dict(values).items():
            if k not in idx:
                raise KeyError(f"Unknown IOAT code: {k}")
            try:
                v = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Value for {k} is not a number: {val!r}") from None
            if not np.isfinite(v):
                raise ValueError(f"Value for {k} is missing or not finite")
            if v < 0:
                raise ValueError(f"Value for {k} is negative ({v}); inputs must be zero or positive")
            out[idx[k]] += v
        return out

    def bridge(self, spend_pp: np.ndarray, kind: str = "procurement") -> dict:
        """Convert spend at purchasers' prices into first-round demand on UK industries
        at basic prices. Returns the UK demand vector and a reconciliation of where the
        rest of the spend went (margins re-routed, product taxes, imports)."""
        s = self.s
        spend = spend_pp * self.in_leontief
        excluded = float((spend_pp - spend).sum())
        m = self.margin_rate if s.remove_margins else 0.0
        t = self.purchase_tax_rate if s.remove_product_taxes else 0.0
        mu = self.mu_proc if kind == "procurement" else self.mu_capex
        mu = np.minimum(1.0, mu * (1 + s.import_uplift)) if s.remove_imports else 0.0

        basic = spend * (1 - m - t)
        margins = float((spend * m).sum())
        taxes = float((spend * t).sum())
        imports = basic * mu
        uk = basic - imports
        idx = {c: i for i, c in enumerate(self.d.codes)}
        for code, share in s.margin_destination.items():
            uk[idx[code]] += margins * share
        return {
            "uk_demand": uk,
            "gross": float(spend_pp.sum()),
            "excluded_no_leontief": excluded,
            "product_taxes": taxes,
            "imports": float(imports.sum()),
            "uk_demand_total": float(uk.sum()),
        }

    # ---- core calculation ----------------------------------------------------------
    def run(self, in_house=None, procurement=None, capital=None, in_house_pay=None,
            direct_fte=None, mode: int = 2) -> "Results":
        """
        in_house     the organisation's own UK activity, £m of output by IOAT industry, valued
                     at the industry's average GVA, wage and FTE ratios
        in_house_pay the organisation's own UK staff costs, £m by IOAT industry (mode 2 only):
                     direct GVA = pay x (1 + direct_margin_on_pay); direct FTE from the
                     industry's FTE per £m of compensation; bought-in costs go in procurement
        direct_fte   reported direct FTE by IOAT industry (for example from payroll days);
                     replaces the modelled direct FTE in that industry
        procurement  UK operating purchases, £m at purchasers' prices by supplying industry
        capital      UK capital spend, £m at purchasers' prices by supplying industry
        mode 1       in-house activity treated as final demand for its industry; the supply
                     chain follows the ONS average technology (reproduces ONS Type I effects).
                     procurement is rejected in this mode; capital spend is added
        mode 2       the supply chain is driven by the reported purchases; in-house activity
                     carries the direct effect only
        """
        y = self.vec(in_house) * self.in_leontief
        b = self.bridge(self.vec(procurement), "procurement")
        k = self.bridge(self.vec(capital), "capital")

        if mode == 1:
            if b["gross"] > 0:
                raise ValueError(
                    "procurement is a mode 2 input. In mode 1 the supply chain of in-house activity "
                    "follows the ONS average for its industry, so adding reported purchases would count "
                    "it twice. Use mode 2 to drive the supply chain from reported purchases.")
            p = self.A @ y + k["uk_demand"]
        elif mode == 2:
            p = b["uk_demand"] + k["uk_demand"]
        else:
            raise ValueError("mode must be 1 or 2")

        pay = self.vec(in_house_pay) * self.in_leontief
        if mode == 1 and pay.any():
            raise ValueError("in_house_pay is a mode 2 input")
        with np.errstate(divide="ignore", invalid="ignore"):
            fte_per_m_pay = np.where(self.w > 0, self.e / self.w, 0.0)
        pay_gva = pay * (1 + self.s.direct_margin_on_pay)

        out_direct = y + pay_gva            # pay-basis entries: output reported equal to GVA
        out_indirect = self.L1 @ p
        W_direct = float(self.w @ y + pay.sum())

        if self.s.include_induced:
            n = self.n
            out_typeII = self.L2[:n, :n] @ p + self.L2[:n, n] * W_direct
            out_induced = out_typeII - out_indirect
        else:
            out_induced = np.zeros(self.n)

        frame = pd.DataFrame(index=self.d.codes)
        frame["industry"] = self.d.names
        pi = self.s.price_index
        for label, o in (("direct", out_direct), ("indirect", out_indirect), ("induced", out_induced)):
            frame[f"output_{label}"] = o
            frame[f"gva_{label}"] = self.v * o
            frame[f"coe_{label}"] = self.w * o
            frame[f"gos_{label}"] = self.g * o
            frame[f"ptax_{label}"] = self.t_prod * o
            frame[f"fte_{label}"] = self.e * o / pi
        frame["gva_direct"] = self.v * y + pay_gva
        frame["coe_direct"] = self.w * y + pay
        frame["gos_direct"] = self.g * y + pay * self.s.direct_margin_on_pay
        frame["ptax_direct"] = self.t_prod * y
        frame["fte_direct"] = (self.e * y + fte_per_m_pay * pay) / pi
        if direct_fte:
            reported = self.vec(direct_fte)
            frame.loc[reported > 0, "fte_direct"] = reported[reported > 0]
        return Results(frame, p, b, k, mode, self.s)

    # ---- closed-form check for induced effects -------------------------------------
    def induced_closed_form(self, p: np.ndarray, W_direct: float) -> np.ndarray:
        u = self.L1 @ self.s_hh
        W1 = W_direct + float(self.w @ (self.L1 @ p))
        kap = self.s.kappa
        return u * kap * W1 / (1 - kap * float(self.w @ u))

    # ---- validation ----------------------------------------------------------------
    def validate(self, tol: float = 1e-6) -> pd.DataFrame:
        d, rows = self.d, []

        def add(name, value, tolerance=tol, info=False):
            status = "info" if info else ("PASS" if abs(value) <= tolerance else "FAIL")
            rows.append({"check": name, "value": value, "status": status})

        bb = d.bluebook
        add("Total output equals Blue Book 2025 (£m)",
            d.output.sum() - bb["Output at basic prices, all industries (£m)"], 0.5)
        add("Total GVA equals Blue Book 2025 (£m)",
            d.gva.sum() - bb["Gross value added at basic prices, all industries (£m)"], 0.5)
        add("Total compensation equals Blue Book 2025 (£m)",
            d.coe.sum() - bb["Compensation of employees, all industries (£m)"], 0.5)
        add("Row total use equals column output, max abs gap (£m)",
            np.abs(d.total_use.values - d.output.values).max(), 0.5)
        add("GVA equals compensation plus operating surplus plus production taxes, max abs gap (£m)",
            np.abs(d.coe.values + d.gos.values + d.ptax.values - d.gva.values).max(), 0.5)
        add("Inputs at purchasers' prices plus GVA equal output, max abs gap (£m)",
            np.abs(d.inputs_pp.values + d.gva.values - d.output.values).max(), 0.5)

        keep = self.in_leontief.astype(bool)
        ons_L = d.ons_leontief.values
        add("L1 against ONS Leontief inverse, max abs element difference",
            np.abs(self.L1[:, keep] - ons_L[:, keep]).max())
        add("Type I output multipliers against ONS, max abs difference",
            np.abs(self.L1.sum(axis=0)[keep] - d.ons_effects["output"].values[keep]).max())
        add("Type I GVA effects against ONS, max abs difference",
            np.abs((self.v @ self.L1)[keep] - d.ons_effects["gva"].values[keep]).max())
        add("Type I compensation effects against ONS, max abs difference",
            np.abs((self.w @ self.L1)[keep]
                   - d.ons_effects["compensation_of_employees"].values[keep]).max())
        add("Type I FTE effects against ONS, max abs difference",
            np.abs((self.e @ self.L1)[keep] - d.ons_fte["type1_fte_effect"].values[keep]).max())

        # Identities on an arbitrary demand vector
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 10, self.n) * self.in_leontief
        add("Leontief identity: (I - A) L1 p = p, max abs (£m)",
            np.abs((np.eye(self.n) - self.A) @ (self.L1 @ p) - p).max())
        n = self.n
        z = self.L2 @ np.append(p, 0.0)
        add("Type II identity: x - A x - c h = p, max abs (£m)",
            np.abs(z[:n] - self.A @ z[:n] - self.c * z[n] - p).max())
        W = 5.0
        L2_route = self.L2[:n, :n] @ p + self.L2[:n, n] * W - self.L1 @ p
        add("Induced output: L2 route equals closed form, max abs (£m)",
            np.abs(L2_route - self.induced_closed_form(p, W)).max())

        # Mode 1 reproduces ONS for a single industry (film and TV production)
        r = FootprintModel(Settings(include_induced=False), d).run(in_house={"J59 & J60": 1.0}, mode=1)
        j = d.codes.index("J59 & J60")
        add("Mode 1: £1 of J59 & J60 final demand reproduces ONS Type I GVA effect",
            r.table[["gva_direct", "gva_indirect"]].values.sum() - d.ons_effects["gva"].values[j])

        # Bridge conserves value
        spend = rng.uniform(0, 5, self.n)
        bb_ = self.bridge(spend)
        add("Price bridge conserves spend (£m)",
            bb_["gross"] - bb_["excluded_no_leontief"] - bb_["product_taxes"]
            - bb_["imports"] - bb_["uk_demand_total"])
        add("Implied whole-economy FTE from ONS coefficients", float(self.e @ d.output.values), info=True)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class Results:
    table: pd.DataFrame
    first_round: np.ndarray
    proc_bridge: dict
    capex_bridge: dict
    mode: int
    settings: Settings

    def summary(self) -> pd.DataFrame:
        t = self.table
        rows = {}
        for measure, unit in (("output", "£m"), ("gva", "£m"), ("coe", "£m"), ("gos", "£m"),
                          ("ptax", "£m"), ("fte", "FTE")):
            d_, i_, n_ = (t[f"{measure}_{k}"].sum() for k in ("direct", "indirect", "induced"))
            rows[f"{measure} ({unit})"] = {
                "direct": d_, "indirect": i_, "induced": n_,
                "type_I_total": d_ + i_, "type_II_total": d_ + i_ + n_,
            }
        return pd.DataFrame(rows).T

    def leakages(self) -> pd.DataFrame:
        rows = []
        for name, b in (("procurement", self.proc_bridge), ("capital", self.capex_bridge)):
            rows.append({
                "spend": name, "gross_pp": b["gross"], "uk_first_round_bp": b["uk_demand_total"],
                "imports": b["imports"], "product_taxes": b["product_taxes"],
                "excluded_no_leontief": b["excluded_no_leontief"],
            })
        return pd.DataFrame(rows).set_index("spend")

    def by_industry(self, top: int | None = None, measure: str = "gva") -> pd.DataFrame:
        t = self.table.copy()
        t[f"{measure}_total"] = t[[f"{measure}_{k}" for k in ("direct", "indirect", "induced")]].sum(axis=1)
        t = t.sort_values(f"{measure}_total", ascending=False)
        return t.head(top) if top else t


# --------------------------------------------------------------------------------------
# Scenario input
# --------------------------------------------------------------------------------------
def read_spend_file(path: Path) -> dict:
    """CSV with columns: ioat_code, spend_type (in_house | in_house_pay | procurement | capital),
    gbp_m. Lines starting with # are comments."""
    df = pd.read_csv(path, comment="#")
    missing = {"ioat_code", "spend_type", "gbp_m"} - set(df.columns)
    if missing:
        raise ValueError(f"Spending file is missing columns: {sorted(missing)}")
    df["gbp_m"] = pd.to_numeric(df["gbp_m"], errors="coerce")
    bad = df[df["gbp_m"].isna() | (df["gbp_m"] < 0)]
    if len(bad):
        raise ValueError(f"gbp_m must be a number of zero or more; check rows {list(bad.index + 1)}")
    out = {}
    for kind in ("in_house", "in_house_pay", "procurement", "capital", "direct_fte"):
        sub = df[df["spend_type"] == kind]
        out[kind] = sub.groupby("ioat_code")["gbp_m"].sum().to_dict()
    unknown = set(df["spend_type"]) - set(out)
    if unknown:
        raise ValueError(f"Unknown spend_type values: {unknown}")
    return out


def _cli():
    ap = argparse.ArgumentParser(description="UK input-output footprint model (ONS IOAT 2023)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="reproduce ONS published Type I results and run identities")
    r = sub.add_parser("run", help="run a spending file")
    r.add_argument("spend_file", type=Path)
    r.add_argument("--mode", type=int, default=2, choices=[1, 2])
    r.add_argument("--type1", action="store_true", help="exclude induced effects")
    args = ap.parse_args()

    pd.set_option("display.width", 140, "display.float_format", "{:,.2f}".format)
    if args.cmd == "validate":
        v = FootprintModel().validate()
        v["value"] = v["value"].map(lambda x: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.1e}")
        print(v.to_string(index=False))
    else:
        m = FootprintModel(Settings(include_induced=not args.type1))
        res = m.run(**read_spend_file(args.spend_file), mode=args.mode)
        print(res.summary())
        print()
        print(res.leakages())


if __name__ == "__main__":
    _cli()
