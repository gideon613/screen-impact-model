"""Checks that the model reproduces ONS published results, keeps its accounting identities, and that
the credit, receipts, slate and welfare calculations behave as documented.
Run with: python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from fiscal import (avec_relief, break_even_additionality, exchequer_receipts, load_config,  # noqa: E402
                    net_fiscal_impact)
from io_footprint import FootprintModel, Settings, load_data  # noqa: E402
from slate import default_settings, load_mapping, prepare, run_slate, summary_table  # noqa: E402
from welfare import Membership, levy_incidence, profit_max_pass_through  # noqa: E402

SLATE = ROOT / "examples" / "illustrative_slate.csv"
PAYROLL = ROOT / "examples" / "illustrative_payroll.csv"


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def model(data):
    return FootprintModel(Settings(), data)


@pytest.fixture(scope="module")
def cfg():
    return load_config("uk")


# ---- engine ---------------------------------------------------------------------------
def test_all_validation_checks_pass(model):
    checks = model.validate()
    failed = checks[checks["status"] == "FAIL"]
    assert failed.empty, failed.to_string()


@pytest.mark.parametrize("code", ["J59 & J60", "R90", "C21", "F41, F42  & F43", "Q86"])
def test_mode1_reproduces_ons_type1_gva_and_fte(data, code):
    m = FootprintModel(Settings(include_induced=False), data)
    res = m.run(in_house={code: 1.0}, mode=1).summary()
    j = data.codes.index(code)
    assert res.loc["gva (£m)", "type_I_total"] == pytest.approx(data.ons_effects["gva"].iloc[j], abs=1e-9)
    assert res.loc["fte (FTE)", "type_I_total"] == pytest.approx(data.ons_fte["type1_fte_effect"].iloc[j], abs=1e-6)


def test_gva_splits_into_wages_surplus_and_taxes(model):
    s = model.run(in_house_pay={"J59 & J60": 10.0}, procurement={"N77": 4.0}, mode=2).summary()
    parts = s.loc["coe (£m)"] + s.loc["gos (£m)"] + s.loc["ptax (£m)"]
    assert np.allclose(parts.values, s.loc["gva (£m)"].values)


def test_results_scale_linearly(model):
    a = model.run(in_house={"J59 & J60": 10.0}, mode=1).summary()
    b = model.run(in_house={"J59 & J60": 20.0}, mode=1).summary()
    assert np.allclose(b.values, 2 * a.values)


def test_mode1_rejects_procurement(model):
    with pytest.raises(ValueError, match="mode 2 input"):
        model.run(in_house={"J59 & J60": 10.0}, procurement={"N77": 4.0}, mode=1)


def test_mode2_supply_chain_follows_reported_purchases(model):
    t = model.run(procurement={"N77": 4.0}, mode=2).table
    assert t["output_indirect"].sum() > 4.0 * (1 - model.margin_rate.max())  # first round plus later rounds
    j = model.d.codes.index("N77")
    assert t["output_indirect"].iloc[j] > 0


@pytest.mark.parametrize("bad", [{"N77": -1.0}, {"N77": float("nan")}, {"N77": "abc"}, {"XYZ": 1.0}])
def test_bad_inputs_are_rejected(model, bad):
    with pytest.raises((ValueError, KeyError)):
        model.run(procurement=bad, mode=2)


def test_product_tax_rate_comes_from_purchasing_industry(data):
    m = FootprintModel(Settings(purchaser_industry="J59 & J60"), data)
    assert m.purchase_tax_rate == pytest.approx(57 / 41130)  # ONS IOAT 2023, film and TV column
    b = m.bridge(m.vec({"N77": 10.0}))
    assert b["product_taxes"] == pytest.approx(10.0 * m.purchase_tax_rate)


def test_owner_occupier_housing_receives_no_demand(model):
    res = model.run(procurement={"L68A": 5.0}, mode=2)
    assert res.table[["output_indirect", "output_induced"]].values.sum() == pytest.approx(0.0)
    assert res.leakages().loc["procurement", "excluded_no_leontief"] == pytest.approx(5.0)


def test_in_house_pay_sets_direct_gva_and_drives_induced(model):
    t = model.run(in_house_pay={"J59 & J60": 10.0}, mode=2).table
    assert t["gva_direct"].sum() == pytest.approx(10.0)
    assert t["output_indirect"].sum() == pytest.approx(0.0)
    assert np.allclose(t["output_induced"].values, model.induced_closed_form(np.zeros(model.n), 10.0))


def test_reported_direct_fte_overrides_model(model):
    t = model.run(in_house_pay={"J59 & J60": 10.0}, direct_fte={"J59 & J60": 150.0}, mode=2).table
    assert t["fte_direct"].sum() == pytest.approx(150.0)


# ---- tax credit --------------------------------------------------------------------------
def test_avec_without_cap(cfg):
    r = avec_relief("hetv", total_core=10.0, uk_core=7.0, uk_vfx=0.0, cfg=cfg)
    assert r.gross_credit == pytest.approx(0.34 * 7.0)
    assert r.net_cost == pytest.approx(0.75 * r.gross_credit)


def test_avec_cap_and_vfx_follow_crec061410(cfg):
    # cap reduction 0.2 against VFX of 1.0: additional credit = 0.39 x 1.0 - 0.34 x (1.0 - 0.2) = 0.118
    r = avec_relief("film", total_core=10.0, uk_core=8.2, uk_vfx=1.0, cfg=cfg)
    assert r.cap_reduction == pytest.approx(0.2)
    assert r.vfx_additional_credit == pytest.approx(0.118)
    assert r.gross_credit == pytest.approx(0.34 * (8.2 - 1.0) + 0.39 * 1.0)


def test_animation_rate_and_unscripted_ineligible(cfg):
    assert avec_relief("animation", 10.0, 5.0, 0.0, cfg).gross_credit == pytest.approx(0.39 * 5.0)
    assert avec_relief("unscripted", 10.0, 10.0, 0.0, cfg).gross_credit == 0.0


def test_net_fiscal_impact_and_break_even():
    assert net_fiscal_impact(30.0, 20.0, 1.0)["Net fiscal impact"] == pytest.approx(10.0)
    be = break_even_additionality(30.0, 20.0)
    assert net_fiscal_impact(30.0, 20.0, be)["Net fiscal impact"] == pytest.approx(0.0)
    assert break_even_additionality(30.0, 20.0, displacement=0.25) > be
    with pytest.raises(ValueError):
        net_fiscal_impact(30.0, 20.0, 1.2)


def test_labour_receipts_use_tax_wedge(model, cfg):
    res = model.run(in_house_pay={"J59 & J60": 10.0}, mode=2)
    rec = exchequer_receipts(res.table, model, cfg, include_induced=False)
    assert rec["Income tax and NICs"] == pytest.approx(0.312 * 10.0)


# ---- slate --------------------------------------------------------------------------------
def test_slate_bridges_and_totals_reconcile():
    prods, _, ctx = run_slate(SLATE)
    for p in prods:
        b = p.bridge
        assert b["Budget"] == pytest.approx(b["Spent outside the UK"] + b["UK spend"])
        assert b["UK spend"] == pytest.approx(b["of which own cast and crew pay"] + b["of which purchases from UK suppliers"])
        assert b["of which purchases from UK suppliers"] == pytest.approx(
            b["Imported content of UK purchases"] + b["Non-deductible product taxes on UK purchases"]
            + b["Purchases with no UK supply chain (imputed rent)"]
            + b["First-round demand on UK industries (basic prices)"])
        assert p.receipts["Product taxes on purchases"] == pytest.approx(
            b["Non-deductible product taxes on UK purchases"])
    t = summary_table(prods, ctx["model"])
    body = t.drop(index="Slate total")
    for col in ("budget", "gva_total", "fte_total", "net_cost_of_relief"):
        assert t.loc["Slate total", col] == pytest.approx(body[col].sum())


def test_slate_input_problems_are_all_reported(cfg):
    raw = pd.read_csv(SLATE, comment="#")
    raw.loc[0, "location"] = "Gotham"
    raw.loc[1, "gbp_m"] = -2.0
    raw.loc[2, "format"] = "documentary"
    raw.loc[3, "cost_category"] = "Craft services"
    with pytest.raises(ValueError) as err:
        prepare(raw, load_mapping(), cfg["production_formats"])
    msg = str(err.value)
    for expected in ("Gotham", "gbp_m", "format must be one of", "Craft services"):
        assert expected in msg


def test_payroll_must_refer_to_uk_jobs_on_known_productions():
    bad = pd.DataFrame({"production_id": ["P1", "P9"], "location": ["Outside UK", "London"], "direct_fte": [10, 5]})
    with pytest.raises(ValueError) as err:
        run_slate(SLATE, payroll=bad)
    assert "UK jobs only" in str(err.value) and "P9" in str(err.value)


def test_reported_jobs_replace_modelled_direct_jobs_only():
    modelled, _, _ = run_slate(SLATE)
    reported, _, _ = run_slate(SLATE, payroll=PAYROLL)
    pay = pd.read_csv(PAYROLL, comment="#").groupby("production_id")["direct_fte"].sum()
    for a, b in zip(modelled, reported):
        sa, sb = a.results.summary(), b.results.summary()
        assert b.fte_source == "reported"
        assert sb.loc["fte (FTE)", "direct"] == pytest.approx(pay[b.production_id])
        assert sb.loc["fte (FTE)", ["indirect", "induced"]].values == pytest.approx(
            sa.loc["fte (FTE)", ["indirect", "induced"]].values)
        assert sb.loc["gva (£m)"].values == pytest.approx(sa.loc["gva (£m)"].values)


def test_regional_table_holds_spending_and_direct_figures_only():
    prods, _, _ = run_slate(SLATE, payroll=PAYROLL)
    for p in prods:
        assert list(p.by_region.columns) == ["uk_spend", "own_cast_and_crew_pay", "reported_direct_fte"]
        assert p.by_region["uk_spend"].sum() == pytest.approx(p.bridge["UK spend"])
        assert p.by_region["own_cast_and_crew_pay"].sum() == pytest.approx(p.bridge["of which own cast and crew pay"])


# ---- welfare ------------------------------------------------------------------------------
def test_consumer_surplus_linear_demand():
    m = Membership(members_m=10, monthly_price=10, elasticity=2.0, marginal_cost_share=0.0)
    assert m.consumer_surplus_m() == pytest.approx(12 * 10 * 10 / 4)


def test_levy_incidence_identities():
    m = Membership(10, 10, 1.2, 0.0)
    absorbed = levy_incidence(m, 0.05, 0.0)
    assert absorbed["borne_by_service_m"] == pytest.approx(absorbed["levy_raised_m"])
    assert absorbed["excess_burden_m"] == pytest.approx(0.0, abs=1e-9)
    full = levy_incidence(m, 0.05, 1.0)
    assert full["borne_by_members_m"] + full["borne_by_service_m"] == pytest.approx(
        full["levy_raised_m"] + full["excess_burden_m"])


def test_marginal_cost_from_pricing_condition():
    m = Membership(10, 10, 1.25)
    assert m.marginal_cost_share == pytest.approx(0.2)
    assert Membership(10, 10, 0.8).marginal_cost_share == 0.0


def test_profit_maximising_pass_through_minimises_the_service_loss():
    m = Membership(10, 10, 1.2)
    best = profit_max_pass_through(m, 0.05)
    assert best == pytest.approx((1 - 1 / 1.2) / (2 * 0.95))
    loss = lambda pt: levy_incidence(m, 0.05, pt)["borne_by_service_m"]
    grid = np.linspace(0, 1, 201)
    assert loss(best) <= min(loss(pt) for pt in grid) + 1e-9
