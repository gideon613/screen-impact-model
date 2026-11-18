"""
Screen production economic impact explorer: a browser front end for the model in this repository.

    streamlit run app.py

Every figure is calculated live by the Python engine (io_footprint, slate, fiscal, welfare) from the
inputs on screen. The production, payroll and membership figures are illustrative.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fiscal import break_even_additionality, load_config, net_fiscal_impact
from io_footprint import FootprintModel, load_data
from slate import (OUTSIDE_UK, SLATE_COLUMNS, UK_LOCATIONS, default_settings, load_mapping, run_slate,
                   summary_table)
from welfare import Membership, levy_incidence, profit_max_pass_through

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"

# Chart tokens (validated categorical palette, light surface)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C1, C2, C3, C4, NEUTRAL, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781", "#ffffff"

st.set_page_config(page_title="Screen production economic impact explorer", layout="wide")


# --------------------------------------------------------------------------------------
# Cached reference data
# --------------------------------------------------------------------------------------
@st.cache_resource
def ons_data():
    return load_data()


@st.cache_resource
def base_config():
    return load_config("uk")


@st.cache_data
def example(name: str) -> pd.DataFrame:
    return pd.read_csv(EXAMPLES / name, comment="#")


@st.cache_data
def mapping() -> pd.DataFrame:
    return load_mapping()


@st.cache_data
def validation_table() -> pd.DataFrame:
    return FootprintModel(default_settings(base_config()), ons_data()).validate()


@st.cache_data
def test_count() -> int:
    """Number of tests pytest collects; falls back to counting test functions."""
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
                             capture_output=True, text=True, cwd=ROOT, timeout=60).stdout
        return int(out.strip().splitlines()[-1].split()[0])
    except Exception:
        names = []
        for f in (ROOT / "tests").glob("test_*.py"):
            names += [n.name for n in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
                      if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        return len(names)


def repo_url() -> str | None:
    url = os.environ.get("REPO_URL")
    if not url:
        try:
            url = st.secrets.get("REPO_URL")
        except Exception:
            url = None
    if not url:
        try:
            url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True,
                                 text=True, timeout=3).stdout.strip()
        except Exception:
            url = None
    if url and url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url.removesuffix(".git") if url and url.startswith("http") else None


INDUSTRY_SHORT = {
    "J59 & J60": "Film, TV, music and broadcasting", "L68BXL683": "Real estate services",
    "F41, F42  & F43": "Construction", "G47": "Retail", "G46": "Wholesale", "N77": "Equipment rental",
    "R90": "Creative and performing arts", "I56": "Food and drink service", "I55": "Accommodation",
    "K64": "Financial services", "K65.1-2 & K65.3": "Insurance", "K66": "Financial support services",
    "M691": "Legal services", "M692": "Accounting", "N78": "Employment agencies", "N80": "Security",
    "J61": "Telecoms", "J62": "IT services", "H493T495": "Land transport", "H51": "Air transport",
    "C32": "Other manufacturing", "C14": "Clothing", "D351": "Electricity", "Q86": "Health",
    "L68A": "Owner-occupied housing", "O84": "Public administration", "P85": "Education",
    "H52": "Warehousing and transport support", "M70": "Head offices and consultancy",
    "M73": "Advertising and market research", "G45": "Motor trade",
}


def gbp(x: float, dp: int = 1) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}£{abs(x):,.{dp}f}m"


def short(name: str) -> str:
    return name.split(" (")[0]


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _axis(title=None, fmt=None, **kw):
    if fmt is not None:
        kw["format"] = fmt
    return alt.Axis(title=title, labelColor=INK2, titleColor=INK2, gridColor=GRID, domainColor="#c3c2b7",
                    tickColor="#c3c2b7", labelFontSize=12, titleFontSize=12, **kw)


def stacked_bars(df: pd.DataFrame, cat: str, part: str, value: str, order: list, colours: list,
                 x_title: str, height: int = 220, fmt: str = ",.1f") -> alt.Chart:
    df = df.copy()
    df["_order"] = df[part].map({p: i for i, p in enumerate(order)})
    bars = alt.Chart(df).mark_bar(stroke=SURFACE, strokeWidth=2, height={"band": 0.6}).encode(
        y=alt.Y(f"{cat}:N", sort=None, axis=_axis(None, labelLimit=260, domain=False, ticks=False)),
        x=alt.X(f"sum({value}):Q", axis=_axis(x_title)),
        color=alt.Color(f"{part}:N", scale=alt.Scale(domain=order, range=colours),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK2, labelFontSize=12)),
        order=alt.Order("_order:Q"),
        tooltip=[alt.Tooltip(f"{cat}:N", title=""), alt.Tooltip(f"{part}:N", title=""),
                 alt.Tooltip(f"{value}:Q", title="£m", format=fmt)],
    )
    totals = df.groupby(cat, sort=False)[value].sum().reset_index()
    labels = alt.Chart(totals).mark_text(align="left", dx=6, fontWeight="bold", color=INK, fontSize=12).encode(
        y=alt.Y(f"{cat}:N", sort=None), x=alt.X(f"{value}:Q"), text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + labels).properties(height=height).configure_view(stroke=None)


def simple_bars(df: pd.DataFrame, cat: str, value: str, x_title: str, colour: str = C1,
                height: int = 260, fmt: str = ",.1f", tooltip_extra=None) -> alt.Chart:
    tips = [alt.Tooltip(f"{cat}:N", title=""), alt.Tooltip(f"{value}:Q", title=x_title, format=fmt)]
    tips += tooltip_extra or []
    bars = alt.Chart(df).mark_bar(color=colour, cornerRadiusEnd=4, height={"band": 0.6}).encode(
        y=alt.Y(f"{cat}:N", sort="-x", axis=_axis(None, labelLimit=300, domain=False, ticks=False)),
        x=alt.X(f"{value}:Q", axis=_axis(x_title)), tooltip=tips)
    labels = bars.mark_text(align="left", dx=6, color=INK2, fontSize=12).encode(
        text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + labels).properties(height=height).configure_view(stroke=None)


# --------------------------------------------------------------------------------------
# Sidebar: data and settings
# --------------------------------------------------------------------------------------
cfg0 = base_config()
DATA = ons_data()

with st.sidebar:
    st.header("Inputs")
    source = st.radio("Production data", ["Illustrative slate", "Upload cost reports"], index=0,
                      help="Upload a CSV laid out like the illustrative slate. Download it below as a template.")
    slate_df = example("illustrative_slate.csv")
    payroll_df = example("illustrative_payroll.csv")
    if source == "Upload cost reports":
        up = st.file_uploader("Cost-report lines (CSV)", type="csv")
        up_pay = st.file_uploader("Payroll jobs (CSV, optional)", type="csv")
        if up is None:
            st.info("Upload a cost-report file to replace the illustrative slate.")
        else:
            slate_df = pd.read_csv(up, comment="#")
            payroll_df = pd.read_csv(up_pay, comment="#") if up_pay is not None else None
    st.download_button("Download the slate template", example("illustrative_slate.csv").to_csv(index=False),
                       "slate_template.csv", "text/csv", width="stretch")
    st.download_button("Download the payroll template", example("illustrative_payroll.csv").to_csv(index=False),
                       "payroll_template.csv", "text/csv", width="stretch")

    st.header("Model settings")
    use_payroll = st.toggle("Use direct jobs reported in payroll", value=payroll_df is not None,
                            disabled=payroll_df is None,
                            help="Replaces the modelled direct jobs with the jobs each production reported. "
                                 "Off: direct jobs are modelled from pay at the film and TV industry average.")
    induced = st.toggle("Include household spending effects (induced)", value=True,
                        help="Type II: activity supported when wages earned on the productions and in their "
                             "supply chains are spent. Off gives Type I results.")
    tax_wedge = st.slider("Tax wedge on labour costs", 25.0, 35.0, 100 * cfg0["household_closure"]["tax_wedge"],
                          0.1, format="%.1f%%", help="Income tax and National Insurance as a share of labour "
                                                    "costs. OECD Taxing Wages 2025: 31.2% in 2023, 29.4% in 2024.") / 100
    saving = st.slider("Household saving ratio", 2.0, 12.0, 100 * cfg0["household_closure"]["saving_ratio"], 0.1,
                       format="%.1f%%", help="ONS Blue Book 2025: 6.2% in 2023, 10.1% in 2024.") / 100
    import_adj = st.slider("Change to ONS import shares", -25, 50, 0, 5, format="%+d%%",
                           help="Proportional change to the ONS import share of each product bought.") / 100
    with st.expander("Sources"):
        st.caption("ONS Input-Output Analytical Tables 2023; ONS Blue Book 2025; OECD Taxing Wages 2025; "
                   "HMRC Annual Report and Accounts 2023 to 2024; BFI guidance on the Audio-Visual "
                   "Expenditure Credit; HMRC Creative Industries Expenditure Credit Manual CREC061410.")

cfg = copy.deepcopy(cfg0)
cfg["exchequer"]["labour_tax_rate_on_compensation"] = tax_wedge
settings = default_settings(cfg, tax_wedge=tax_wedge, saving_ratio=saving, include_induced=induced,
                            import_uplift=import_adj)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("Screen production economic impact explorer")
url = repo_url()
st.markdown(
    "A working model of the UK economy that turns the records a commissioning business already keeps "
    "(production cost reports, payroll and membership data) into evidence on economic contribution, "
    "the tax credit and the cost of a levy. Every figure is calculated live from the inputs on screen. "
    "**All production and membership figures are illustrative** and do not describe any real production, "
    "studio or service.")
st.caption("Built by Gideon Morrison-Wood on ONS Input-Output Analytical Tables 2023."
           + (f" [Source code and method]({url})" if url else ""))

tab_econ, tab_fiscal, tab_subs, tab_valid, tab_method = st.tabs(
    ["Economic contribution", "Tax credit and public finances", "Subscriptions and levies",
     "Model validation", "Method and limitations"])

# --------------------------------------------------------------------------------------
# Scenario controls (top of the first tab, applied before the model runs)
# --------------------------------------------------------------------------------------
if slate_df is None or slate_df.empty or "production_id" not in slate_df:
    st.error("The cost-report file has no production_id column. Use the slate template as a guide.")
    st.stop()
slate_df["production_id"] = slate_df["production_id"].astype(str)
names = slate_df.drop_duplicates("production_id").set_index("production_id")["production"].astype(str)

with tab_econ:
    left, right = st.columns([1, 1])
    focus = left.selectbox("Show results for", ["Whole slate"] + names.tolist())
    focus_ids = list(names.index) if focus == "Whole slate" else [names.index[names.tolist().index(focus)]]
    with right:
        reshore = st.slider("Bring spending now outside the UK into the UK", 0, 100, 0, 5, format="%d%%",
                            help="Moves this share of the selected productions' non-UK spend to a UK location. "
                                 "The tax credit is capped at 80% of core expenditure, so the credit rises "
                                 "by less than the spend.")
        dest = st.selectbox("Into", UK_LOCATIONS, index=UK_LOCATIONS.index("South East"),
                            disabled=reshore == 0)
    with st.expander("Edit a production's cost report"):
        ed_id = st.selectbox("Production", names.index, format_func=lambda i: names[i], key="ed_pick")
        cats = mapping()["cost_category"].tolist()
        base_lines = slate_df[slate_df["production_id"] == ed_id][["cost_category", "location", "gbp_m"]]
        edited = st.data_editor(
            base_lines.reset_index(drop=True), num_rows="dynamic", key=f"editor_{ed_id}_{source}",
            column_config={
                "cost_category": st.column_config.SelectboxColumn("Cost category", options=cats, required=True),
                "location": st.column_config.SelectboxColumn("Where incurred", options=UK_LOCATIONS + [OUTSIDE_UK],
                                                             required=True),
                "gbp_m": st.column_config.NumberColumn("£m", min_value=0.0, step=0.1, format="%.2f",
                                                       required=True),
            }, width="stretch")
        st.caption("Change an amount or a location, or add a line, and every result updates. "
                   "Reported payroll jobs stay at their reported values; switch off reported jobs in the "
                   "sidebar to see modelled direct jobs respond to changes in pay.")


def apply_scenario(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    meta = df.drop_duplicates("production_id").set_index("production_id")[["production", "format"]]
    ed = edited.dropna(how="all").copy()
    ed["production_id"] = ed_id
    ed["production"] = meta.loc[ed_id, "production"]
    ed["format"] = meta.loc[ed_id, "format"]
    df = pd.concat([df[df["production_id"] != ed_id], ed[SLATE_COLUMNS]], ignore_index=True)
    if reshore:
        move = df["production_id"].isin(focus_ids) & (df["location"] == OUTSIDE_UK)
        moved = df[move].copy()
        moved["gbp_m"] = pd.to_numeric(moved["gbp_m"], errors="coerce") * reshore / 100
        moved["location"] = dest
        df.loc[move, "gbp_m"] = pd.to_numeric(df.loc[move, "gbp_m"], errors="coerce") * (1 - reshore / 100)
        df = pd.concat([df, moved], ignore_index=True)
    return df


def run(slate: pd.DataFrame, payroll: pd.DataFrame | None):
    return run_slate(slate, settings, data=DATA, cfg=cfg, payroll=payroll if use_payroll else None)


try:
    scen_slate = apply_scenario(slate_df)
    prods, _, ctx = run(scen_slate, payroll_df)
    base_prods, _, _ = run(slate_df, payroll_df)
except (ValueError, KeyError) as err:
    st.error(f"The model could not run on these inputs.\n\n{err}")
    st.stop()

model = ctx["model"]
summ = summary_table(prods, model)
scenario_on = reshore > 0 or not edited.reset_index(drop=True).equals(base_lines.reset_index(drop=True))


def pick(ps, ids):
    return [p for p in ps if p.production_id in ids]


def totals(ps) -> dict:
    s = sum(p.results.summary() for p in ps)
    elig = [p for p in ps if p.relief.eligible]
    return {
        "budget": sum(p.bridge["Budget"] for p in ps), "uk_spend": sum(p.bridge["UK spend"] for p in ps),
        "gva": s.loc["gva (£m)", "type_II_total"], "fte": s.loc["fte (FTE)", "type_II_total"],
        "fte_direct": s.loc["fte (FTE)", "direct"], "wages": s.loc["coe (£m)", "type_II_total"],
        "gos": s.loc["gos (£m)", "type_II_total"],
        "credit": sum(p.relief.gross_credit for p in elig), "net_cost": sum(p.relief.net_cost for p in elig),
        "receipts_elig": sum(p.receipts["Total receipts"] for p in elig), "n_elig": len(elig),
    }


sel, base_sel = pick(prods, focus_ids), pick(base_prods, focus_ids)
if not sel or not base_sel:
    st.warning("The selected production has no cost lines. Add a line in the cost-report editor.")
    st.stop()
T, B = totals(sel), totals(base_sel)


def delta(key, fmt):
    if not scenario_on:
        return None
    d = T[key] - B[key]
    return None if abs(d) < 1e-9 else fmt(d)


# --------------------------------------------------------------------------------------
# Tab 1: economic contribution
# --------------------------------------------------------------------------------------
with tab_econ:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UK spend", gbp(T["uk_spend"]), delta("uk_spend", lambda d: f"{d:+,.1f}m"), border=True,
              help=f"Out of total budgets of {gbp(T['budget'])}.")
    m2.metric("UK GVA supported", gbp(T["gva"]), delta("gva", lambda d: f"{d:+,.1f}m"), border=True,
              help="Gross value added: wages, operating surplus and taxes on production, across direct, "
                   "supply-chain and (if switched on) household-spending effects.")
    m3.metric("Jobs supported (FTE)", f"{T['fte']:,.0f}", delta("fte", lambda d: f"{d:+,.0f}"), border=True,
              help=f"{T['fte_direct']:,.0f} of them on the productions themselves.")
    m4.metric("Wages paid", gbp(T["wages"]), delta("wages", lambda d: f"{d:+,.1f}m"), border=True,
              help=f"Operating surplus earned by UK businesses: {gbp(T['gos'])}.")
    if scenario_on:
        st.caption("Changes are shown against the inputs as loaded.")

    st.subheader("GVA supported, by production and effect")
    rows = []
    for p in sel:
        s = p.results.summary()
        for eff in ("direct", "indirect", "induced"):
            rows.append({"Production": short(p.production), "Effect": eff.capitalize(),
                         "gva": s.loc["gva (£m)", eff]})
    gdf = pd.DataFrame(rows)
    gdf = gdf[gdf["Effect"].isin(["Direct", "Indirect"] + (["Induced"] if induced else []))]
    st.altair_chart(stacked_bars(gdf, "Production", "Effect", "gva", ["Direct", "Indirect", "Induced"],
                                 [C1, C2, C3], "£m", height=70 + 52 * len(sel)), width="stretch")
    st.caption("Direct: the productions' own cast and crew pay. Indirect: UK suppliers and their suppliers. "
               "Induced: UK activity supported when the wages earned are spent.")

    st.subheader("From budget to UK demand")
    bridge = pd.DataFrame({short(p.production): p.bridge for p in sel})
    if len(sel) > 1:
        bridge["Total"] = bridge.sum(axis=1)
    st.dataframe(bridge.style.format("{:,.1f}"), width="stretch", height=36 * (len(bridge) + 1) + 3)
    st.caption("£m. Imported content and product taxes leave the UK supply chain; distribution margins "
               "are passed to wholesale. Each step reconciles to the budget.")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Where the money was spent")
        reg = pd.concat([p.by_region for p in sel]).groupby(level=0).sum()
        reg = reg.reset_index().rename(columns={"location": "Nation or region"})
        st.altair_chart(simple_bars(reg, "Nation or region", "uk_spend", "UK spend, £m", colour=C1,
                                    height=40 + 34 * len(reg), tooltip_extra=[
                                        alt.Tooltip("own_cast_and_crew_pay:Q", title="Own cast and crew pay, £m",
                                                    format=",.1f"),
                                        alt.Tooltip("reported_direct_fte:Q", title="Reported direct jobs",
                                                    format=",.0f")]), width="stretch")
        st.caption("Spending, pay and reported jobs by where they occurred. The ONS tables cover the UK as a "
                   "whole, so supply-chain and household-spending effects are not estimated by region.")
    with c_right:
        st.subheader("Industries where the GVA is supported")
        ind = sum(p.results.table.drop(columns="industry") for p in sel)
        ind["gva_total"] = ind[["gva_direct", "gva_indirect"] + (["gva_induced"] if induced else [])].sum(axis=1)
        ind["Industry"] = [INDUSTRY_SHORT.get(c, DATA.names[c]) for c in ind.index]
        top = ind.sort_values("gva_total", ascending=False).head(8)
        st.altair_chart(simple_bars(top, "Industry", "gva_total", "GVA supported, £m", colour=C1, height=300),
                        width="stretch")
        st.caption("Top eight industries, all effects. Film and TV production includes the productions' own pay.")
    st.download_button("Download results by production (CSV)", summary_table(prods, model).to_csv(),
                       "slate_summary.csv", "text/csv")

# --------------------------------------------------------------------------------------
# Tab 2: tax credit and public finances
# --------------------------------------------------------------------------------------
with tab_fiscal:
    elig = [p for p in sel if p.relief.eligible]
    st.markdown(
        "The Audio-Visual Expenditure Credit is calculated on UK core expenditure, capped at 80% of total "
        "core expenditure: 34% for film and high-end TV, 39% for animation and children's TV, and 39% on UK "
        "visual effects, which are exempt from the cap. The credit is taxable, so its net cost to the "
        "Exchequer is 75% of the gross credit. Unscripted entertainment does not qualify.")
    if not elig:
        st.info("None of the selected productions qualifies for the credit.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Gross credit", gbp(T["credit"]), delta("credit", lambda d: f"{d:+,.1f}m"), border=True)
        f2.metric("Net cost to the Exchequer", gbp(T["net_cost"]), delta("net_cost", lambda d: f"{d:+,.1f}m"),
                  delta_color="inverse", border=True)
        f3.metric("Gross tax receipts supported", gbp(T["receipts_elig"]),
                  delta("receipts_elig", lambda d: f"{d:+,.1f}m"), border=True,
                  help="Tax paid on all the activity the eligible productions support, whether or not the "
                       "credit caused it.")
        f4.metric("Gross receipts per £1 of net relief", f"£{T['receipts_elig'] / T['net_cost']:.2f}", border=True)

        rows = []
        for p in elig:
            rows.append({"Production": short(p.production), "Item": "Net cost of the credit", "£m": p.relief.net_cost})
            for k, lab in (("Income tax and NICs", "Income tax and NICs"),
                           ("VAT and duties on household spending", "VAT and duties"),
                           ("Corporation tax", "Corporation tax"),
                           ("Taxes on production", "Other taxes")):
                v = p.receipts[k] + (p.receipts["Product taxes on purchases"] if k == "Taxes on production" else 0)
                rows.append({"Production": short(p.production), "Item": lab, "£m": v})
        fd = pd.DataFrame(rows)
        fd["Bar"] = np.where(fd["Item"] == "Net cost of the credit", "Net cost of the credit", "Gross receipts")
        order = ["Net cost of the credit", "Income tax and NICs", "VAT and duties", "Corporation tax", "Other taxes"]
        fd["_o"] = fd["Item"].map({k: i for i, k in enumerate(order)})
        chart = alt.Chart(fd).mark_bar(stroke=SURFACE, strokeWidth=2).encode(
            y=alt.Y("Bar:N", sort=["Net cost of the credit", "Gross receipts"],
                    axis=_axis(None, domain=False, ticks=False, labelLimit=200)),
            x=alt.X("sum(£m):Q", axis=_axis("£m")),
            color=alt.Color("Item:N", scale=alt.Scale(domain=order, range=[NEUTRAL, C1, C2, C3, C4]),
                            legend=alt.Legend(orient="top", title=None, labelColor=INK2, labelFontSize=12)),
            order=alt.Order("_o:Q"),
            row=alt.Row("Production:N", sort=[short(p.production) for p in elig],
                        header=alt.Header(labelAngle=0, labelAlign="left", labelColor=INK, labelFontSize=13,
                                          labelFontWeight="bold", title=None)),
            tooltip=["Production", "Item", alt.Tooltip("£m:Q", format=",.2f")],
        ).properties(height=70, width=680).configure_view(stroke=None)
        st.altair_chart(chart, width="content")

        st.subheader("From gross receipts to the effect on the public finances")
        st.markdown(
            "Gross receipts measure the contribution of the activity. They include tax on any activity that "
            "would have taken place in the UK without the credit. What the credit itself does to the public "
            "finances depends on two shares that come from evidence on how location decisions respond to the "
            "credit: **additionality**, the share of the activity that would not have taken place in the UK "
            "without it, and **displacement**, the share of that additional activity that draws on workers and "
            "resources that would otherwise have been employed elsewhere in the UK. The credit is paid on "
            "every eligible production, so its cost does not fall with additionality.")
        a1, a2 = st.columns(2)
        add = a1.slider("Additionality", 0, 100, 50, 5, format="%d%%") / 100
        disp = a2.slider("Displacement", 0, 75, 0, 5, format="%d%%") / 100
        nf = net_fiscal_impact(T["receipts_elig"], T["net_cost"], add, disp)
        be = break_even_additionality(T["receipts_elig"], T["net_cost"], disp)
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Receipts caused by the credit", gbp(nf["Receipts caused by the credit"]), border=True)
        n2.metric("Net cost of the credit", gbp(nf["Net cost of relief"]), border=True)
        n3.metric("Net fiscal impact", gbp(nf["Net fiscal impact"]), border=True,
                  help="Receipts caused by the credit less its net cost.")
        n4.metric("Break-even additionality", f"{be:.0%}" if be <= 1 else "above 100%", border=True,
                  help="The additionality at which receipts caused by the credit equal its net cost, "
                       "at the displacement chosen.")
        grid = pd.DataFrame({"Additionality": np.linspace(0, 1, 101)})
        grid["Net fiscal impact, £m"] = [net_fiscal_impact(T["receipts_elig"], T["net_cost"], a, disp)["Net fiscal impact"]
                                         for a in grid["Additionality"]]
        line = alt.Chart(grid).mark_line(color=C1, strokeWidth=2).encode(
            x=alt.X("Additionality:Q", axis=_axis("Additionality", fmt="%", values=[i / 10 for i in range(11)])),
            y=alt.Y("Net fiscal impact, £m:Q", axis=_axis("Net fiscal impact, £m")),
            tooltip=[alt.Tooltip("Additionality:Q", format=".0%"),
                     alt.Tooltip("Net fiscal impact, £m:Q", format=",.1f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
        point = alt.Chart(pd.DataFrame({"Additionality": [add], "Net fiscal impact, £m": [nf["Net fiscal impact"]]})
                          ).mark_point(filled=True, size=110, color=C1, stroke=SURFACE, strokeWidth=2).encode(
            x="Additionality:Q", y="Net fiscal impact, £m:Q")
        st.altair_chart((line + zero + point).properties(height=280).configure_view(stroke=None), width="stretch")

        st.subheader("Credit and receipts by production")
        tbl = summ.loc[[p.production for p in sel], ["format", "uk_core_expenditure", "qualifying_within_cap",
                                                     "uk_vfx", "gross_credit", "net_cost_of_relief",
                                                     "gross_receipts", "gross_receipts_per_1_relief",
                                                     "break_even_additionality"]]
        tbl.index = [short(i) for i in tbl.index]
        tbl.index.name = "Production"
        tbl.columns = ["Format", "UK core", "Within cap", "UK VFX", "Gross credit", "Net cost",
                       "Gross receipts", "Per £1", "Break-even"]
        tbl[tbl.columns[1:]] = tbl[tbl.columns[1:]].astype(float)
        tbl["Per £1"] = tbl["Per £1"].map(lambda v: f"£{v:.2f}" if pd.notna(v) else "not eligible")
        tbl["Break-even"] = tbl["Break-even"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "not eligible")
        st.dataframe(tbl.style.format({c: "{:,.1f}" for c in tbl.columns[1:7]}), width="stretch")
        st.caption("£m. Per £1: gross receipts per £1 of net cost. Break-even: the additionality at which "
                   "receipts caused by the credit equal its net cost, with no displacement. Single-period "
                   "simplification; assumes every eligible production is certified as British. Receipts use "
                   "average effective tax rates.")

# --------------------------------------------------------------------------------------
# Tab 3: subscriptions and levies
# --------------------------------------------------------------------------------------
with tab_subs:
    mem0 = json.loads((EXAMPLES / "illustrative_membership.json").read_text(encoding="utf-8"))
    st.markdown(
        "A subscription service holds the inputs for this view in its own records: members, revenue per "
        "member, and the response of sign-ups and cancellations to past price changes. The view estimates "
        "the value members get above what they pay, and who would bear a levy on subscription revenue. "
        "Demand is linear through the current price and membership, the conservative choice for the value "
        "to members.")
    s1, s2, s3 = st.columns(3)
    members = s1.number_input("Members (millions)", 0.1, 100.0, float(mem0["members_m"]), 0.5)
    price = s2.number_input("Average monthly revenue per member (£)", 1.0, 50.0, float(mem0["monthly_price"]), 0.5)
    elast = s3.number_input("Price elasticity of membership", 0.1, 5.0, float(mem0["elasticity"]), 0.1,
                            help="Percentage fall in members for a 1% rise in price.")
    s4, s5, s6 = st.columns(3)
    levy_rate = s4.slider("Levy on subscription revenue", 0.0, 10.0, 100 * float(mem0["levy_rate"]), 0.5,
                          format="%.1f%%") / 100
    mc_mode = s5.radio("Cost of serving one more member", ["Consistent with the current price", "Set directly"],
                       help="Consistent: a service that sets its price to maximise profit has a margin of "
                            "1/elasticity over the cost of serving one more member.")
    mc_share = None
    if mc_mode == "Set directly":
        mc_share = s5.slider("As a share of price", 0, 90, 10, 5, format="%d%%") / 100
    try:
        mem = Membership(members, price, elast, mc_share)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    pt_opt = profit_max_pass_through(mem, levy_rate)
    pt_mode = s6.radio("Share of the levy passed on in prices", ["Profit-maximising", "Set directly"])
    pt = pt_opt if pt_mode == "Profit-maximising" else s6.slider("Pass-through", 0, 100, 100, 5, format="%d%%") / 100
    li = levy_incidence(mem, levy_rate, pt)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Value to members", gbp(mem.consumer_surplus_m(), 0), border=True,
              help="Annual consumer surplus: value to members above what they pay, linear demand.")
    k2.metric("Levy raised a year", gbp(li["levy_raised_m"]), border=True)
    k3.metric("Borne by members", gbp(li["borne_by_members_m"]), border=True,
              help=f"Price rises {pt * levy_rate:.2%}; about {li['members_lost_m'] * 1000:,.0f} thousand members leave.")
    k4.metric("Borne by the service", gbp(li["borne_by_service_m"]), border=True)
    st.caption(f"Marginal cost per member: {mem.marginal_cost_share:.0%} of price. Pass-through used: {pt:.0%}"
               f" (a profit-maximising service would pass on {pt_opt:.0%}). Loss to members and the service "
               f"beyond the levy raised: {gbp(li['excess_burden_m'])} a year.")

    curve = []
    for x in np.linspace(0, 1, 41):
        r = levy_incidence(mem, levy_rate, x)
        curve += [{"Pass-through": x, "Series": "Borne by members", "£m": r["borne_by_members_m"]},
                  {"Pass-through": x, "Series": "Borne by the service", "£m": r["borne_by_service_m"]},
                  {"Pass-through": x, "Series": "Total borne", "£m": r["borne_by_members_m"] + r["borne_by_service_m"]},
                  {"Pass-through": x, "Series": "Levy raised", "£m": r["levy_raised_m"]}]
    cd = pd.DataFrame(curve)
    lines = alt.Chart(cd).mark_line(strokeWidth=2).encode(
        x=alt.X("Pass-through:Q", axis=_axis("Share of the levy passed on in prices", fmt="%",
                                             values=[i / 10 for i in range(11)])),
        y=alt.Y("£m:Q", axis=_axis("£m a year")),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Borne by members", "Borne by the service",
                                                            "Total borne", "Levy raised"],
                                                    range=[C1, C2, C3, INK2]),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK2, labelFontSize=12)),
        strokeDash=alt.condition(alt.datum.Series == "Levy raised", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("Pass-through:Q", format=".0%"), "Series:N", alt.Tooltip("£m:Q", format=",.1f")])
    rule = alt.Chart(pd.DataFrame({"x": [pt_opt]})).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(x="x:Q")
    rule_lab = alt.Chart(pd.DataFrame({"x": [pt_opt], "t": ["profit-maximising"]})).mark_text(
        align="left", dx=5, dy=-6, color=INK2, fontSize=11).encode(x="x:Q", y=alt.value(12), text="t:N")
    st.altair_chart((lines + rule + rule_lab).properties(height=300).configure_view(stroke=None), width="stretch")
    st.caption("A levy on revenue takes the same share of revenue at every price, so it gives a service with low "
               "running costs little reason to raise its price. Passing on more than the profit-maximising "
               "share loses members whose revenue exceeded the cost of serving them.")

# --------------------------------------------------------------------------------------
# Tab 4: model validation
# --------------------------------------------------------------------------------------
with tab_valid:
    st.markdown(
        "The supply-chain engine is built from the ONS Input-Output Analytical Tables for 2023 and reproduces "
        "the ONS published Type I results for all 104 industries. The checks below run on the tables "
        "loaded by this app.")
    v = validation_table().copy()
    v["Result"] = v["value"].map(lambda x: f"{x:,.0f}" if abs(x) >= 1 else ("exact" if x == 0 else f"{x:.1e}"))
    v = v.rename(columns={"check": "Check", "status": "Status"})[["Check", "Result", "Status"]]
    st.dataframe(v, hide_index=True, width="stretch", height=36 * (len(v) + 1) + 3)

    st.subheader("Check any industry against ONS")
    code = st.selectbox("Industry", DATA.codes, index=DATA.codes.index("J59 & J60"),
                        format_func=lambda c: f"{c}: {DATA.names[c]}")
    if code == "L68A":
        st.info("Owner-occupiers' housing (imputed rent) has no ONS Leontief column, so it is excluded.")
    else:
        mt = FootprintModel(default_settings(cfg0, include_induced=False), DATA)
        r = mt.run(in_house={code: 1.0}, mode=1).summary()
        j = DATA.codes.index(code)
        cmp = pd.DataFrame({
            "Measure (per £1 of final demand)": ["Output", "GVA", "Compensation of employees", "FTE jobs per £m"],
            "This model": [r.loc["output (£m)", "type_I_total"], r.loc["gva (£m)", "type_I_total"],
                           r.loc["coe (£m)", "type_I_total"], r.loc["fte (FTE)", "type_I_total"]],
            "ONS published": [DATA.ons_effects["output"].iloc[j], DATA.ons_effects["gva"].iloc[j],
                              DATA.ons_effects["compensation_of_employees"].iloc[j],
                              DATA.ons_fte["type1_fte_effect"].iloc[j]],
        })
        cmp["Difference"] = cmp["This model"] - cmp["ONS published"]
        st.dataframe(cmp.style.format({"This model": "{:.6f}", "ONS published": "{:.6f}", "Difference": "{:.1e}"}),
                     hide_index=True, width="stretch")

    st.subheader("Automated tests")
    st.markdown(f"The repository holds {test_count()} automated tests covering the ONS reproduction, "
                "accounting identities, input checks, the credit against HMRC's worked method, the spend "
                "bridge, reported jobs, the net fiscal calculation, the levy incidence and this app. They run "
                "on every change to the code.")
    if st.button("Run the test suite now"):
        with st.spinner("Running tests"):
            out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", str(ROOT / "tests")],
                                 capture_output=True, text=True, cwd=ROOT, timeout=300)
        (st.success if out.returncode == 0 else st.error)(out.stdout.strip().splitlines()[-1] if out.stdout else
                                                          "pytest did not run")
        with st.expander("Test output"):
            st.code(out.stdout[-4000:] or out.stderr[-4000:])

# --------------------------------------------------------------------------------------
# Tab 5: method and limitations
# --------------------------------------------------------------------------------------
with tab_method:
    st.markdown("""
**Supply chain.** The ONS domestic use table records what each of 104 industries buys from every other UK
industry. The Leontief inverse of the input coefficients gives the UK output needed across every round of
the supply chain to meet £1 of demand. Industry ratios convert output into GVA, wages, operating surplus,
taxes on production and jobs.

**From spending to UK demand.** Cost-report lines are mapped to ONS industries. Spending is at purchasers'
prices, net of recoverable VAT. The bridge removes distribution margins (passed to wholesale), the
non-deductible product taxes the film and TV industry pays on its purchases (its own ONS rate, 0.14%, counted
as a tax receipt) and the imported share of each product. Own cast and crew pay is the direct GVA.

**Household spending.** Induced effects close the model with respect to households. The share of each £1 of
wages spent on UK output comes from the OECD tax wedge and the ONS saving ratio.

**Credit and receipts.** The Audio-Visual Expenditure Credit follows the published rates, the 80% cap and
HMRC's method for visual effects. Receipts apply average effective tax rates to the wages, spending and
operating surplus supported. They are gross; the net fiscal view applies additionality and displacement.

**Members and levies.** Linear demand through current price and membership. The cost of serving one more
member is set from the pricing condition unless entered directly.

**Limitations.** Input-output models assume fixed input proportions and spare capacity, so they describe the
activity associated with spending at today's structure of the economy. Results are contribution estimates
and make no causal claim about the credit without evidence on additionality. The ONS tables are UK-wide, so
regional results are limited to spending, pay and reported jobs. Many crew are freelance and pay a different
mix of taxes from the average employee. The cost-category mapping and the credit flags are indicative and
would be agreed with production accountants. Coefficients describe 2023; inputs from other years need a price
index for the jobs figures. Spillovers such as screen tourism, skills and exports are outside the model.
""")
    if url:
        st.markdown(f"Full method, sources and code: [{url}]({url}). One-page technical and policy note: "
                    f"[technical_note.pdf]({url}/blob/main/docs/technical_note.pdf).")
