"""
Exchequer side of production spending: the cost of the screen tax credit and the tax receipts
supported by the activity, on the same gross basis as the GVA and jobs results.

Relief: Audio-Visual Expenditure Credit (AVEC), single-period simplification.
    Chapter 3 credit   = rate x min(UK core, 80% x total core)
    Additional VFX     = 39% x UK VFX - 34% x max(0, UK VFX - cap reduction)     (CREC061410)
    Net Exchequer cost = gross credit x (1 - corporation tax rate)               (credit is taxable)

Gross receipts supported (indicative, average effective rates):
    labour taxes       = tax wedge x compensation of employees (income tax and NICs)
    taxes on spending  = product tax rate on household consumption x consumer spending of wages
    production taxes   = ONS taxes on production, by industry (business rates etc.); industries
                         where subsidies exceed taxes count as zero. The ONS film and TV
                         industry shows net subsidies of £2.1bn in 2023, consistent with the
                         screen credits, which are costed separately here
    corporation tax    = HMRC CT receipts / UK operating surplus x operating surplus supported
    product taxes      = non-deductible product taxes on the productions' own UK purchases, at the
                         film and TV industry's ONS rate (removed in the price bridge)

Gross and net. Receipts supported are gross: they are the taxes paid on all the activity the
productions support, whether or not the credit caused it. They measure contribution, not impact.
The change in the public finances caused by the credit depends on
    additionality   the share of the activity that would not have taken place in the UK without
                    the credit
    displacement    the share of that additional activity that draws workers and resources away
                    from other UK activity instead of adding to it
    net fiscal impact = gross receipts x additionality x (1 - displacement) - net cost of relief
The credit is paid on every eligible production, so its cost does not fall with additionality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_config(market: str = "uk") -> dict:
    return json.loads((CONFIG_DIR / f"{market}.json").read_text(encoding="utf-8"))


@dataclass
class ReliefResult:
    eligible: bool
    total_core: float
    uk_core: float
    uk_vfx: float
    cap_reduction: float
    qualifying: float
    chapter3_credit: float
    vfx_additional_credit: float
    gross_credit: float
    net_cost: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def avec_relief(fmt: str, total_core: float, uk_core: float, uk_vfx: float, cfg: dict) -> ReliefResult:
    inc = cfg["screen_incentive"]
    if fmt not in inc["eligible_formats"]:
        return ReliefResult(False, total_core, uk_core, uk_vfx, 0, 0, 0, 0, 0, 0)
    rate = inc["credit_rate_by_format"][fmt]
    cap = inc["cap_share_of_core_expenditure"] * total_core
    qualifying = min(uk_core, cap)
    cap_reduction = max(0.0, uk_core - cap)
    ch3 = rate * qualifying
    vfx_add = 0.0
    if fmt in inc["vfx_formats"] and uk_vfx > 0:
        adjusted = rate * max(0.0, uk_vfx - cap_reduction)
        vfx_add = inc["vfx_credit_rate"] * uk_vfx - adjusted
    gross = ch3 + vfx_add
    net = gross * (1 - inc["credit_is_taxable_at"])
    return ReliefResult(True, total_core, uk_core, uk_vfx, cap_reduction, qualifying, ch3, vfx_add, gross, net)


def exchequer_receipts(table: pd.DataFrame, model, cfg: dict, include_induced: bool = True,
                       purchase_taxes: float = 0.0) -> dict:
    """Gross tax receipts supported by the activity in a results table, £m. purchase_taxes is the
    non-deductible product tax removed in the price bridge, which the Exchequer receives."""
    ex = cfg["exchequer"]
    effects = ["direct", "indirect"] + (["induced"] if include_induced else [])
    coe = sum(table[f"coe_{k}"].sum() for k in effects)
    gos = sum(table[f"gos_{k}"].sum() for k in effects)
    t_pos = np.clip(model.t_prod, 0, None)
    ptax = sum(float((table[f"output_{k}"].values * t_pos).sum()) for k in effects)

    d = model.d
    keep = model.in_leontief.astype(bool)
    ct_rate_on_gos = ex["corporation_tax_receipts_gbp_m"] / float(d.gos.values[keep].sum())

    labour = ex["labour_tax_rate_on_compensation"] * coe
    # consumer spending out of all wages supported (in a Type I view, only wages actually earned)
    spending = model.s.kappa * coe if include_induced else 0.0
    taxes_on_spending = model.hh_product_tax_rate * spending
    corp = ct_rate_on_gos * gos
    return {
        "Income tax and NICs": labour,
        "VAT and duties on household spending": taxes_on_spending,
        "Taxes on production": ptax,
        "Corporation tax": corp,
        "Product taxes on purchases": purchase_taxes,
        "Total receipts": labour + taxes_on_spending + ptax + corp + purchase_taxes,
        "_ct_rate_on_operating_surplus": ct_rate_on_gos,
    }


def net_fiscal_impact(gross_receipts: float, relief_net_cost: float, additionality: float,
                      displacement: float = 0.0) -> dict:
    """Change in the public finances caused by the credit, £m, for a stated additionality and
    displacement (both shares between 0 and 1)."""
    for name, v in (("additionality", additionality), ("displacement", displacement)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    additional = gross_receipts * additionality * (1 - displacement)
    return {
        "Gross receipts supported": gross_receipts,
        "Receipts caused by the credit": additional,
        "Net cost of relief": relief_net_cost,
        "Net fiscal impact": additional - relief_net_cost,
    }


def break_even_additionality(gross_receipts: float, relief_net_cost: float, displacement: float = 0.0) -> float:
    """Additionality at which receipts caused by the credit equal its net cost. Above 1 means the
    credit cannot pay for itself at that displacement."""
    base = gross_receipts * (1 - displacement)
    return relief_net_cost / base if base > 0 else np.nan


def fiscal_balance(receipts: dict, relief_net_cost: float, displacement: float = 0.0) -> dict:
    total = receipts["Total receipts"]
    return {
        "Gross receipts supported (£m)": total,
        "Net cost of relief (£m)": relief_net_cost,
        "Gross receipts per £1 of relief": total / relief_net_cost if relief_net_cost else np.nan,
        "Break-even additionality": break_even_additionality(total, relief_net_cost, displacement),
    }
