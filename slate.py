"""
Production slate pipeline: from cost-report lines to economic and fiscal results.

Input is a long table with one row per production and cost category (the top sheet of a production
cost report), the UK nation or region where the cost was incurred, and the amount. A mapping file
assigns each cost category to an ONS industry, a spend type (own staff costs or purchases) and its
status for the screen tax credit (core expenditure, visual effects). The same mapping serves every
production, so a slate of any size runs in one pass and every production is treated alike.

An optional payroll file gives the direct FTE jobs each production reported, by UK nation or region.
Where it is supplied, reported jobs replace the modelled direct jobs for that production.

For each production the pipeline reports:
    spend bridge     budget, spend outside the UK, UK spend, imports, product taxes, first-round UK demand
    economic impact  GVA, wages, operating surplus, taxes on production and FTE jobs,
                     split into direct, indirect and induced
    relief           AVEC qualifying expenditure, gross credit and net cost to the Exchequer
    receipts         gross tax receipts supported, and break-even additionality
    by region        UK spend, own cast and crew pay and reported direct jobs by where the cost was
                     incurred. These are spending and direct figures only: the ONS tables are for
                     the UK as a whole, so the model does not estimate regional multiplier effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fiscal import avec_relief, exchequer_receipts, load_config
from io_footprint import FootprintModel, Settings, load_data

ROOT = Path(__file__).resolve().parent
OUTSIDE_UK = "Outside UK"
UK_LOCATIONS = ["North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands",
                "East of England", "London", "South East", "South West", "Wales", "Scotland",
                "Northern Ireland"]
SLATE_COLUMNS = ["production_id", "production", "format", "cost_category", "location", "gbp_m"]
PAYROLL_COLUMNS = ["production_id", "location", "direct_fte"]
SPEND_TYPES = {"in_house_pay", "procurement", "capital"}


# --------------------------------------------------------------------------------------
# Loading and input checks
# --------------------------------------------------------------------------------------
def _read(src) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    return pd.read_csv(src, comment="#")


def load_slate(path) -> pd.DataFrame:
    return _read(path)


def load_mapping(path: Path = ROOT / "mappings" / "cost_categories.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _rows(mask: pd.Series) -> str:
    rows = list(mask[mask].index + 1)
    return ", ".join(map(str, rows[:8])) + (" ..." if len(rows) > 8 else "")


def check_mapping(mapping: pd.DataFrame, codes: list) -> list[str]:
    errors = []
    if mapping["cost_category"].duplicated().any():
        errors.append("Mapping lists a cost category more than once: "
                      f"{sorted(mapping.loc[mapping['cost_category'].duplicated(), 'cost_category'])}")
    bad_code = ~mapping["ioat_code"].isin(codes)
    if bad_code.any():
        errors.append(f"Mapping uses IOAT codes not in the ONS tables: {sorted(mapping.loc[bad_code, 'ioat_code'])}")
    bad_type = ~mapping["spend_type"].isin(SPEND_TYPES)
    if bad_type.any():
        errors.append(f"Mapping spend_type must be one of {sorted(SPEND_TYPES)}; check rows {_rows(bad_type)}")
    for flag in ("core", "vfx"):
        bad = ~mapping[flag].isin(["Y", "N"])
        if bad.any():
            errors.append(f"Mapping column {flag} must be Y or N; check rows {_rows(bad)}")
    vfx_not_core = mapping["vfx"].eq("Y") & ~mapping["core"].eq("Y")
    if vfx_not_core.any():
        errors.append("Visual effects categories must also be core expenditure: "
                      f"{sorted(mapping.loc[vfx_not_core, 'cost_category'])}")
    return errors


def check_slate(slate: pd.DataFrame, mapping: pd.DataFrame, formats: list) -> list[str]:
    missing = [c for c in SLATE_COLUMNS if c not in slate.columns]
    if missing:
        return [f"Slate is missing columns: {missing}. Expected {SLATE_COLUMNS}"]
    if slate.empty:
        return ["Slate has no rows"]
    errors = []
    amount = pd.to_numeric(slate["gbp_m"], errors="coerce")
    bad = amount.isna() | ~np.isfinite(amount.fillna(0)) | (amount < 0)
    if bad.any():
        errors.append(f"gbp_m must be a number of zero or more; check rows {_rows(bad)}")
    blank = slate[["production_id", "production", "format", "cost_category", "location"]].isna().any(axis=1)
    if blank.any():
        errors.append(f"Blank entries in rows {_rows(blank)}")
    bad_fmt = ~slate["format"].isin(formats)
    if bad_fmt.any():
        errors.append(f"format must be one of {formats}; check rows {_rows(bad_fmt)}")
    bad_loc = ~slate["location"].isin(UK_LOCATIONS + [OUTSIDE_UK])
    if bad_loc.any():
        errors.append(f"location must be a UK nation or ITL1 region, or '{OUTSIDE_UK}'; "
                      f"unrecognised: {sorted(slate.loc[bad_loc, 'location'].astype(str).unique())}")
    unknown = sorted(set(slate["cost_category"].dropna()) - set(mapping["cost_category"]))
    if unknown:
        errors.append(f"Cost categories missing from the mapping: {unknown}")
    per = slate.groupby("production_id").agg(n_fmt=("format", "nunique"), n_name=("production", "nunique"))
    mixed = per[(per["n_fmt"] > 1) | (per["n_name"] > 1)].index.tolist()
    if mixed:
        errors.append(f"Each production_id needs one name and one format; check {mixed}")
    return errors


def check_payroll(payroll: pd.DataFrame, slate: pd.DataFrame) -> list[str]:
    missing = [c for c in PAYROLL_COLUMNS if c not in payroll.columns]
    if missing:
        return [f"Payroll file is missing columns: {missing}. Expected {PAYROLL_COLUMNS}"]
    errors = []
    fte = pd.to_numeric(payroll["direct_fte"], errors="coerce")
    bad = fte.isna() | (fte < 0)
    if bad.any():
        errors.append(f"direct_fte must be a number of zero or more; check rows {_rows(bad)}")
    bad_loc = ~payroll["location"].isin(UK_LOCATIONS)
    if bad_loc.any():
        errors.append("Payroll locations must be UK nations or ITL1 regions (report UK jobs only); "
                      f"check rows {_rows(bad_loc)}")
    unknown = sorted(set(payroll["production_id"].astype(str)) - set(slate["production_id"].astype(str)))
    if unknown:
        errors.append(f"Payroll refers to productions not in the slate: {unknown}")
    return errors


def prepare(slate: pd.DataFrame, mapping: pd.DataFrame, formats: list | None = None,
            codes: list | None = None) -> pd.DataFrame:
    """Check the slate and mapping, then attach the mapping to each cost line. Raises ValueError
    listing every problem found, so a user can fix an input file in one pass."""
    formats = formats or load_config("uk")["production_formats"]
    errors = check_mapping(mapping, codes) if codes is not None else []
    errors += check_slate(slate, mapping, formats)
    if errors:
        raise ValueError("Input problems found:\n- " + "\n- ".join(errors))
    df = slate.merge(mapping, on="cost_category", how="left", validate="many_to_one")
    df["gbp_m"] = pd.to_numeric(df["gbp_m"])
    df["production_id"] = df["production_id"].astype(str)
    df["uk"] = df["location"] != OUTSIDE_UK
    df["core"] = df["core"].eq("Y")
    df["vfx"] = df["vfx"].eq("Y")
    return df


def default_settings(cfg: dict, **overrides) -> Settings:
    """Central-case settings for a market: household closure and the purchasing industry for
    product taxes are taken from the market configuration."""
    base = dict(tax_wedge=cfg["household_closure"]["tax_wedge"],
                saving_ratio=cfg["household_closure"]["saving_ratio"],
                purchaser_industry=cfg.get("purchaser_industry"))
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------------------
# Running productions
# --------------------------------------------------------------------------------------
@dataclass
class ProductionResult:
    production_id: str
    production: str
    fmt: str
    results: object            # io_footprint.Results
    relief: object             # fiscal.ReliefResult
    receipts: dict
    receipts_type1: dict
    bridge: dict
    fte_source: str = "modelled"
    by_region: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_production(lines: pd.DataFrame, model: FootprintModel, cfg: dict,
                   payroll: pd.DataFrame | None = None) -> ProductionResult:
    uk = lines[lines["uk"]]
    pay = uk[uk["spend_type"] == "in_house_pay"].groupby("ioat_code")["gbp_m"].sum().to_dict()
    proc = uk[uk["spend_type"] == "procurement"].groupby("ioat_code")["gbp_m"].sum().to_dict()
    cap = uk[uk["spend_type"] == "capital"].groupby("ioat_code")["gbp_m"].sum().to_dict()

    # Reported direct jobs replace modelled direct jobs, attached to the industries where own pay
    # is recorded, in proportion to pay
    direct_fte, fte_source = None, "modelled"
    if payroll is not None and len(payroll):
        total_fte = float(payroll["direct_fte"].sum())
        total_pay = sum(pay.values())
        weights = {k: v / total_pay for k, v in pay.items()} if total_pay > 0 else {"J59 & J60": 1.0}
        direct_fte = {k: total_fte * w for k, w in weights.items()}
        fte_source = "reported"

    res = model.run(in_house_pay=pay, procurement=proc, capital=cap, direct_fte=direct_fte, mode=2)

    fmt = lines["format"].iloc[0]
    total_core = lines.loc[lines["core"], "gbp_m"].sum()
    uk_core = lines.loc[lines["core"] & lines["uk"], "gbp_m"].sum()
    uk_vfx = lines.loc[lines["core"] & lines["uk"] & lines["vfx"], "gbp_m"].sum()
    relief = avec_relief(fmt, total_core, uk_core, uk_vfx, cfg)

    purchase_taxes = res.proc_bridge["product_taxes"] + res.capex_bridge["product_taxes"]
    receipts = exchequer_receipts(res.table, model, cfg, include_induced=True, purchase_taxes=purchase_taxes)
    receipts_t1 = exchequer_receipts(res.table, model, cfg, include_induced=False, purchase_taxes=purchase_taxes)

    pb, kb = res.proc_bridge, res.capex_bridge
    bridge = {
        "Budget": lines["gbp_m"].sum(),
        "Spent outside the UK": lines.loc[~lines["uk"], "gbp_m"].sum(),
        "UK spend": uk["gbp_m"].sum(),
        "of which own cast and crew pay": sum(pay.values()),
        "of which purchases from UK suppliers": sum(proc.values()) + sum(cap.values()),
        "Imported content of UK purchases": pb["imports"] + kb["imports"],
        "Non-deductible product taxes on UK purchases": purchase_taxes,
        "Purchases with no UK supply chain (imputed rent)": pb["excluded_no_leontief"] + kb["excluded_no_leontief"],
        "First-round demand on UK industries (basic prices)": pb["uk_demand_total"] + kb["uk_demand_total"],
    }

    # Spending and direct figures by nation and region. No regional multipliers: UK tables only.
    reg = uk.groupby("location").agg(uk_spend=("gbp_m", "sum"))
    reg["own_cast_and_crew_pay"] = uk[uk["spend_type"] == "in_house_pay"].groupby("location")["gbp_m"].sum()
    reg["reported_direct_fte"] = (payroll.groupby("location")["direct_fte"].sum()
                                  if payroll is not None and len(payroll) else 0.0)
    reg = reg.fillna(0.0)

    return ProductionResult(str(lines["production_id"].iloc[0]), lines["production"].iloc[0], fmt,
                            res, relief, receipts, receipts_t1, bridge, fte_source, reg)


def run_slate(slate, settings: Settings | None = None, market: str = "uk", data=None,
              cfg: dict | None = None, payroll=None) -> tuple[list[ProductionResult], pd.DataFrame, dict]:
    """slate and payroll may be CSV paths or DataFrames. payroll is optional."""
    cfg = cfg or load_config(market)
    s = settings or default_settings(cfg)
    model = FootprintModel(s, data or load_data())
    raw = load_slate(slate)
    df = prepare(raw, load_mapping(), cfg["production_formats"], model.d.codes)
    pay_df = None
    if payroll is not None:
        pay_df = _read(payroll)
        errors = check_payroll(pay_df, raw)
        if errors:
            raise ValueError("Input problems found:\n- " + "\n- ".join(errors))
        pay_df["production_id"] = pay_df["production_id"].astype(str)
        pay_df["direct_fte"] = pd.to_numeric(pay_df["direct_fte"])
    prods = []
    for pid, g in df.groupby("production_id", sort=True):
        p_pay = pay_df[pay_df["production_id"] == pid] if pay_df is not None else None
        prods.append(run_production(g, model, cfg, p_pay))
    return prods, df, {"model": model, "cfg": cfg}


def summary_table(prods: list[ProductionResult], model: FootprintModel) -> pd.DataFrame:
    rows = []
    for p in prods:
        s = p.results.summary()
        net = p.relief.net_cost
        gva = s.loc["gva (£m)", "type_II_total"]
        rec = p.receipts["Total receipts"]
        rows.append({
            "production": p.production, "format": p.fmt,
            "budget": p.bridge["Budget"], "uk_spend": p.bridge["UK spend"],
            "gva_direct": s.loc["gva (£m)", "direct"], "gva_indirect": s.loc["gva (£m)", "indirect"],
            "gva_induced": s.loc["gva (£m)", "induced"], "gva_total": gva,
            "wages": s.loc["coe (£m)", "type_II_total"], "operating_surplus": s.loc["gos (£m)", "type_II_total"],
            "taxes_on_production": s.loc["ptax (£m)", "type_II_total"],
            "fte_direct": s.loc["fte (FTE)", "direct"], "fte_total": s.loc["fte (FTE)", "type_II_total"],
            "direct_fte_source": p.fte_source,
            "avec_eligible": p.relief.eligible, "uk_core_expenditure": p.relief.uk_core,
            "qualifying_within_cap": p.relief.qualifying, "uk_vfx": p.relief.uk_vfx,
            "gross_credit": p.relief.gross_credit, "net_cost_of_relief": net,
            "gross_receipts": rec,
            "gross_receipts_per_1_relief": rec / net if net else np.nan,
            "break_even_additionality": net / rec if net else np.nan,
            "gross_gva_per_1_relief": gva / net if net else np.nan,
        })
    t = pd.DataFrame(rows).set_index("production")
    tot = t.select_dtypes("number").sum()
    elig = t[t["avec_eligible"] == True]  # noqa: E712  ratios use credit-eligible productions only
    if len(elig):
        tot["gross_receipts_per_1_relief"] = elig["gross_receipts"].sum() / elig["net_cost_of_relief"].sum()
        tot["break_even_additionality"] = elig["net_cost_of_relief"].sum() / elig["gross_receipts"].sum()
        tot["gross_gva_per_1_relief"] = elig["gva_total"].sum() / elig["net_cost_of_relief"].sum()
    else:
        tot[["gross_receipts_per_1_relief", "break_even_additionality", "gross_gva_per_1_relief"]] = np.nan
    t.loc["Slate total"] = tot
    t.loc["Slate total", "format"] = "all"
    t.loc["Slate total", "avec_eligible"] = sum(p.relief.eligible for p in prods)
    srcs = {p.fte_source for p in prods}
    t.loc["Slate total", "direct_fte_source"] = srcs.pop() if len(srcs) == 1 else "mixed"
    return t
