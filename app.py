"""
Screen production economic impact explorer: a browser front end for the model in this repository.

    streamlit run app.py

Every figure is calculated live by the Python engine (io_footprint, slate, fiscal, welfare) from the
inputs on screen. The production, payroll and membership figures are illustrative.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from elasticity import PANEL, event_study
from fiscal import break_even_additionality, load_config, net_fiscal_impact
from io_footprint import load_data
from slate import (OUTSIDE_UK, SLATE_COLUMNS, UK_LOCATIONS, default_settings, load_mapping, run_slate,
                   summary_table)
from welfare import Membership, levy_incidence, profit_max_pass_through

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"

# Chart tokens (validated categorical palette, light surface)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C1, C2, C3, C4, NEUTRAL, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781", "#ffffff"

EFFECTS = {"direct": "On the productions", "indirect": "In supply chains", "induced": "From spending of wages"}
FORMAT_NAMES = {"film": "Film", "hetv": "High-end TV", "animation": "Animation", "childrens": "Children's TV",
                "unscripted": "Unscripted"}
BRIDGE_LABELS = {
    "Budget": "Budget",
    "Spent outside the UK": "Spent outside the UK",
    "UK spend": "Spent in the UK",
    "of which own cast and crew pay": "  of which cast and crew pay",
    "of which purchases from UK suppliers": "  of which bought from UK businesses",
    "Imported content of UK purchases": "Imported content of those purchases",
    "Non-deductible product taxes on UK purchases": "Taxes paid on those purchases",
    "Purchases with no UK supply chain (imputed rent)": "Purchases with no UK supply chain",
    "First-round demand on UK industries (basic prices)": "Received by UK businesses",
}
INDUSTRY_SHORT = {
    "J59 & J60": "Film, TV, music and broadcasting", "L68BXL683": "Real estate services",
    "F41, F42  & F43": "Construction", "G47": "Retail", "G46": "Wholesale", "N77": "Equipment rental",
    "R90": "Creative and performing arts", "I56": "Restaurants and catering", "I55": "Hotels",
    "K64": "Financial services", "K65.1-2 & K65.3": "Insurance", "K66": "Financial support services",
    "M691": "Legal services", "M692": "Accounting", "N78": "Employment agencies", "N80": "Security",
    "J61": "Telecoms", "J62": "IT services", "H493T495": "Land transport", "H51": "Air transport",
    "C32": "Other manufacturing", "C14": "Clothing", "D351": "Electricity", "Q86": "Health",
    "L68A": "Owner-occupied housing", "O84": "Public administration", "P85": "Education",
    "H52": "Warehousing and transport support", "M70": "Head offices and consultancy",
    "M73": "Advertising and market research", "G45": "Motor trade",
}

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
def price_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL, comment="#")


@st.cache_data(show_spinner="Estimating")
def estimate_response(panel: pd.DataFrame):
    return event_study(panel)


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


def gbp(x: float, dp: int = 1) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}£{abs(x):,.{dp}f}m"


def short(name: str) -> str:
    return name.split(" (")[0]


def count_words(n: int) -> str:
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(n, str(n))


def in_short(text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**In short.** {text}")


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _axis(title=None, fmt=None, **kw):
    if fmt is not None:
        kw["format"] = fmt
    return alt.Axis(title=title, labelColor=INK2, titleColor=INK2, gridColor=GRID, domainColor="#c3c2b7",
                    tickColor="#c3c2b7", labelFontSize=12, titleFontSize=12, **kw)


def _legend():
    return alt.Legend(orient="top", title=None, labelColor=INK2, labelFontSize=12, labelLimit=260)


def stacked_bars(df: pd.DataFrame, cat: str, part: str, value: str, order: list, colours: list,
                 x_title: str, height: int = 220, fmt: str = ",.1f") -> alt.Chart:
    df = df.copy()
    df["_order"] = df[part].map({p: i for i, p in enumerate(order)})
    bars = alt.Chart(df).mark_bar(stroke=SURFACE, strokeWidth=2, height={"band": 0.6}).encode(
        y=alt.Y(f"{cat}:N", sort=None, axis=_axis(None, labelLimit=260, domain=False, ticks=False)),
        x=alt.X(f"sum({value}):Q", axis=_axis(x_title)),
        color=alt.Color(f"{part}:N", scale=alt.Scale(domain=order, range=colours), legend=_legend()),
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
                      help="Upload a CSV file laid out like the illustrative slate. Download the templates below.")
    slate_df = example("illustrative_slate.csv")
    payroll_df = example("illustrative_payroll.csv")
    if source == "Upload cost reports":
        up = st.file_uploader("Cost-report lines (CSV)", type="csv")
        up_pay = st.file_uploader("Jobs from payroll (CSV, optional)", type="csv")
        if up is None:
            st.info("Upload a cost-report file to replace the illustrative slate.")
        else:
            slate_df = pd.read_csv(up, comment="#")
            payroll_df = pd.read_csv(up_pay, comment="#") if up_pay is not None else None
    st.download_button("Download the cost-report template", example("illustrative_slate.csv").to_csv(index=False),
                       "slate_template.csv", "text/csv", width="stretch")
    st.download_button("Download the payroll template", example("illustrative_payroll.csv").to_csv(index=False),
                       "payroll_template.csv", "text/csv", width="stretch")

    st.header("Settings")
    use_payroll = st.toggle("Use jobs reported in payroll", value=payroll_df is not None,
                            disabled=payroll_df is None,
                            help="On: jobs on the productions are taken from payroll. Off: they are estimated from "
                                 "pay, at the average pay in film and TV.")
    induced = st.toggle("Count the spending of wages", value=True,
                        help="On: includes the activity supported when people working on the productions and for "
                             "their suppliers spend their pay (induced effects). Off: counts only the productions and "
                             "their supply chains.")
    tax_wedge = st.slider("Income tax and National Insurance, share of pay", 25.0, 35.0,
                          100 * cfg0["household_closure"]["tax_wedge"], 0.1, format="%.1f%%",
                          help="Share of total employment costs paid in income tax and National Insurance. "
                               "OECD: 31.2% in 2023, 29.4% in 2024.") / 100
    saving = st.slider("Share of income households save", 2.0, 12.0, 100 * cfg0["household_closure"]["saving_ratio"],
                       0.1, format="%.1f%%", help="ONS: 6.2% in 2023, 10.1% in 2024.") / 100
    import_adj = st.slider("Change to the imported share of purchases", -25, 50, 0, 5, format="%+d%%",
                           help="Raises or lowers the official share of each product bought that comes from "
                                "abroad.") / 100
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
    "Governments are intetested in answering three related economic questions: what does media production spending "
    "add to the economy, what do the public finances get back for the tax credit provided, and who would "
    "bear any tax or levy applied to subscriptions. This model uses data a typical business would keep internally: "
    "production cost reports, payroll and membership data. The outputs are calculated from the inputs shown on screen. "
    "**All production and membership figures are illustrative and do not describe any real production, studio "
    "or service.**")
st.caption("Built by Gideon Morrison-Wood based on ONS input-output tables for 2023."
           + (f" [Source code and method]({url})" if url else ""))

tab_econ, tab_fiscal, tab_price, tab_subs, tab_method = st.tabs(
    ["Economic contribution", "Tax credit", "Response to price", "Subscriptions and levies", "How it works"])

# --------------------------------------------------------------------------------------
# Scenario controls (top of the first tab, applied before the model runs)
# --------------------------------------------------------------------------------------
if slate_df is None or slate_df.empty or "production_id" not in slate_df:
    st.error("The cost-report file has no production_id column. Use the cost-report template as a guide.")
    st.stop()
slate_df["production_id"] = slate_df["production_id"].astype(str)
names = slate_df.drop_duplicates("production_id").set_index("production_id")["production"].astype(str)

with tab_econ:
    headline_econ = st.container()
    left, right = st.columns([1, 1])
    focus = left.selectbox("Show results for", ["All productions"] + names.tolist())
    focus_ids = list(names.index) if focus == "All productions" else [names.index[names.tolist().index(focus)]]
    with right:
        reshore = st.slider("Bring spending now made abroad into the UK", 0, 100, 0, 5, format="%d%%",
                            help="Moves this share of the selected productions' spending outside the UK to a UK "
                                 "location. The tax credit is capped at 80% of production spending, so it rises by "
                                 "less than the spending.")
        dest = st.selectbox("Move it to", UK_LOCATIONS, index=UK_LOCATIONS.index("South East"),
                            disabled=reshore == 0)
    with st.expander("Edit a production's cost report"):
        ed_id = st.selectbox("Production", names.index, format_func=lambda i: names[i], key="ed_pick")
        cats = mapping()["cost_category"].tolist()
        base_lines = slate_df[slate_df["production_id"] == ed_id][["cost_category", "location", "gbp_m"]]
        edited = st.data_editor(
            base_lines.reset_index(drop=True), num_rows="dynamic", key=f"editor_{ed_id}_{source}",
            column_config={
                "cost_category": st.column_config.SelectboxColumn("Cost category", options=cats, required=True),
                "location": st.column_config.SelectboxColumn("Where spent", options=UK_LOCATIONS + [OUTSIDE_UK],
                                                             required=True),
                "gbp_m": st.column_config.NumberColumn("£m", min_value=0.0, step=0.1, format="%.2f",
                                                       required=True),
            }, width="stretch")
        st.caption("Change an amount or a location, or add a line, and every result updates. Jobs reported in "
                   "payroll stay as reported; switch them off in the sidebar to see jobs on the productions move "
                   "with pay.")


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
subject = (f"The {count_words(len(sel))} productions" if len(sel) > 1
           else f"The {short(sel[0].production).lower()}")


def delta(key, fmt):
    if not scenario_on:
        return None
    d = T[key] - B[key]
    return None if abs(d) < 1e-9 else fmt(d)


# --------------------------------------------------------------------------------------
# Tab 1: economic contribution
# --------------------------------------------------------------------------------------
with headline_econ:
    reach = ("Counting their suppliers and the spending of the wages they paid"
             if induced else "Counting their suppliers")
    in_short(f"{subject} spent {gbp(T['uk_spend'])} in the UK. {reach}, that spending supported "
             f"{gbp(T['gva'])} of UK GDP and {T['fte']:,.0f} full-time jobs, {T['fte_direct']:,.0f} of them on the "
             f"productions themselves.")

with tab_econ:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UK spend", gbp(T["uk_spend"]), delta("uk_spend", lambda d: f"{d:+,.1f}m"), border=True,
              help=f"Out of total budgets of {gbp(T['budget'])}.")
    m2.metric("UK GDP supported", gbp(T["gva"]), delta("gva", lambda d: f"{d:+,.1f}m"), border=True,
              help="Gross value added (GVA): the wages, profits and taxes on production generated on the productions, "
                   "in their supply chains and, if switched on, through the spending of wages.")
    m3.metric("Jobs supported (full-time)", f"{T['fte']:,.0f}", delta("fte", lambda d: f"{d:+,.0f}"),
              border=True, help=f"Full-time equivalent jobs. {T['fte_direct']:,.0f} are on the productions themselves.")
    m4.metric("Wages paid", gbp(T["wages"]), delta("wages", lambda d: f"{d:+,.1f}m"), border=True,
              help=f"Profits earned by UK businesses before the cost of their capital: {gbp(T['gos'])}.")
    if scenario_on:
        st.caption("Changes are shown against the inputs as loaded.")

    st.subheader("Where the contribution to GDP comes from")
    rows = []
    for p in sel:
        s = p.results.summary()
        for eff, lab in EFFECTS.items():
            rows.append({"Production": short(p.production), "Effect": lab, "gva": s.loc["gva (£m)", eff]})
    gdf = pd.DataFrame(rows)
    shown = list(EFFECTS.values()) if induced else list(EFFECTS.values())[:2]
    gdf = gdf[gdf["Effect"].isin(shown)]
    st.altair_chart(stacked_bars(gdf, "Production", "Effect", "gva", list(EFFECTS.values()), [C1, C2, C3],
                                 "£m", height=70 + 52 * len(sel)), width="stretch")
    st.caption("On the productions: cast and crew pay. In supply chains: UK businesses that sell to the "
               "productions, and the businesses they buy from. From spending of wages: activity supported when "
               "people in both groups spend their pay.")

    st.subheader("How the budget reaches UK businesses")
    bridge = pd.DataFrame({short(p.production): p.bridge for p in sel})
    if len(sel) > 1:
        bridge["Total"] = bridge.sum(axis=1)
    bridge = bridge.rename(index=BRIDGE_LABELS)
    st.dataframe(bridge.style.format("{:,.1f}"), width="stretch", height=36 * (len(bridge) + 1) + 3)
    st.caption("£m. Money spent on imports, or paid in tax on purchases, does not reach UK businesses. Shop and "
               "wholesale margins are counted as sales by wholesalers. Each line adds back to the budget.")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Where the money was spent")
        reg = pd.concat([p.by_region for p in sel]).groupby(level=0).sum()
        reg = reg.reset_index().rename(columns={"location": "Nation or region"})
        st.altair_chart(simple_bars(reg, "Nation or region", "uk_spend", "UK spend, £m", colour=C1,
                                    height=40 + 34 * len(reg), tooltip_extra=[
                                        alt.Tooltip("own_cast_and_crew_pay:Q", title="Cast and crew pay, £m",
                                                    format=",.1f"),
                                        alt.Tooltip("reported_direct_fte:Q", title="Jobs reported in payroll",
                                                    format=",.0f")]), width="stretch")
        st.caption("Spending, pay and reported jobs by where they took place. The official tables cover the UK as "
                   "a whole, so the knock-on effects in supply chains and wage spending are not split by region.")
    with c_right:
        st.subheader("Industries that benefit")
        ind = sum(p.results.table.drop(columns="industry") for p in sel)
        ind["gva_total"] = ind[["gva_direct", "gva_indirect"] + (["gva_induced"] if induced else [])].sum(axis=1)
        ind["Industry"] = [INDUSTRY_SHORT.get(c, DATA.names[c]) for c in ind.index]
        top = ind.sort_values("gva_total", ascending=False).head(8)
        st.altair_chart(simple_bars(top, "Industry", "gva_total", "Contribution to GDP, £m", colour=C1,
                                    height=300), width="stretch")
        st.caption("The eight industries with the largest contribution. Film and TV includes the productions' "
                   "own pay.")
    st.download_button("Download results by production (CSV)", summary_table(prods, model).to_csv(),
                       "slate_summary.csv", "text/csv")

# --------------------------------------------------------------------------------------
# Tab 2: tax credit
# --------------------------------------------------------------------------------------
with tab_fiscal:
    elig = [p for p in sel if p.relief.eligible]
    if not elig:
        in_short(f"{subject} would not qualify for the tax credit: unscripted entertainment is outside the scheme.")
    else:
        be0 = break_even_additionality(T["receipts_elig"], T["net_cost"], 0.0)
        names_elig = (f"the {count_words(len(elig))} eligible productions" if len(elig) > 1
                      else f"the {short(elig[0].production).lower()}")
        pays = (f"The credit pays for itself only if at least {be0:.0%} of that activity came to the UK because "
                f"of it." if be0 <= 1 else "The credit could not pay for itself even if all of that activity came "
                                            "to the UK because of it.")
        in_short(f"{names_elig[0].upper() + names_elig[1:]} would claim {gbp(T['credit'])} of tax credit, which costs the "
                 f"Exchequer {gbp(T['net_cost'])} once the credit itself is taxed. The activity they support pays "
                 f"{gbp(T['receipts_elig'])} in tax, or £{T['receipts_elig'] / T['net_cost']:.2f} for every £1 of "
                 f"credit. {pays}")
    st.markdown(
        "The Audio-Visual Expenditure Credit pays 34% of UK production spending on films and high-end TV, and 39% "
        "on animation and children's TV, on no more than 80% of total production spending. UK visual effects "
        "earn 39% and are not subject to the 80% cap. The credit is taxed, so it costs the Exchequer 75% of what "
        "is claimed. Unscripted entertainment does not qualify.")
    if elig:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Credit claimed", gbp(T["credit"]), delta("credit", lambda d: f"{d:+,.1f}m"), border=True)
        f2.metric("Cost of the credit after tax", gbp(T["net_cost"]), delta("net_cost", lambda d: f"{d:+,.1f}m"),
                  delta_color="inverse", border=True)
        f3.metric("Tax paid on the activity", gbp(T["receipts_elig"]),
                  delta("receipts_elig", lambda d: f"{d:+,.1f}m"), border=True,
                  help="Tax on all the activity the eligible productions support, including any that would have "
                       "happened in the UK without the credit.")
        f4.metric("Tax paid per £1 the credit costs", f"£{T['receipts_elig'] / T['net_cost']:.2f}", border=True)

        rows = []
        for p in elig:
            rows.append({"Production": short(p.production), "Item": "Cost of the credit", "£m": p.relief.net_cost})
            for k, lab in (("Income tax and NICs", "Income tax and National Insurance"),
                           ("VAT and duties on household spending", "VAT and duties"),
                           ("Corporation tax", "Corporation tax"),
                           ("Taxes on production", "Other taxes")):
                v = p.receipts[k] + (p.receipts["Product taxes on purchases"] if k == "Taxes on production" else 0)
                rows.append({"Production": short(p.production), "Item": lab, "£m": v})
        fd = pd.DataFrame(rows)
        fd["Bar"] = np.where(fd["Item"] == "Cost of the credit", "Cost of the credit", "Tax paid")
        order = ["Cost of the credit", "Income tax and National Insurance", "VAT and duties", "Corporation tax",
                 "Other taxes"]
        fd["_o"] = fd["Item"].map({k: i for i, k in enumerate(order)})
        chart = alt.Chart(fd).mark_bar(stroke=SURFACE, strokeWidth=2).encode(
            y=alt.Y("Bar:N", sort=["Cost of the credit", "Tax paid"],
                    axis=_axis(None, domain=False, ticks=False, labelLimit=200)),
            x=alt.X("sum(£m):Q", axis=_axis("£m")),
            color=alt.Color("Item:N", scale=alt.Scale(domain=order, range=[NEUTRAL, C1, C2, C3, C4]), legend=_legend()),
            order=alt.Order("_o:Q"),
            row=alt.Row("Production:N", sort=[short(p.production) for p in elig],
                        header=alt.Header(labelAngle=0, labelAlign="left", labelColor=INK, labelFontSize=13,
                                          labelFontWeight="bold", title=None)),
            tooltip=["Production", "Item", alt.Tooltip("£m:Q", format=",.2f")],
        ).properties(height=70, width=680).configure_view(stroke=None)
        st.altair_chart(chart, width="content")

        st.subheader("Does the credit pay for itself?")
        st.markdown(
            "The tax figures above count tax on all the activity the productions support, including any that would "
            "have happened in the UK without the credit. Whether the credit pays for itself depends on two shares. "
            "The first is the share of the activity that came to the UK because of the credit (additionality). The "
            "second is the share of that activity whose crews and suppliers would otherwise have been busy on other "
            "UK work (displacement). The credit is paid on every eligible production, so its cost stays the same "
            "whatever these shares are. Both shares come from evidence on how location decisions respond to the "
            "credit, which the model takes as inputs.")
        a1, a2 = st.columns(2)
        add = a1.slider("Share of activity brought to the UK by the credit", 0, 100, 50, 5, format="%d%%",
                        help="Additionality.") / 100
        disp = a2.slider("Share that would otherwise have gone to other UK work", 0, 75, 0, 5, format="%d%%",
                         help="Displacement.") / 100
        nf = net_fiscal_impact(T["receipts_elig"], T["net_cost"], add, disp)
        be = break_even_additionality(T["receipts_elig"], T["net_cost"], disp)
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Tax brought in by the credit", gbp(nf["Receipts caused by the credit"]), border=True)
        n2.metric("Cost of the credit", gbp(nf["Net cost of relief"]), border=True)
        n3.metric("Gain or loss to the Exchequer", gbp(nf["Net fiscal impact"]), border=True,
                  help="Tax brought in by the credit, less what the credit costs after tax.")
        n4.metric("Credit pays for itself above", f"{be:.0%}" if be <= 1 else "never", border=True,
                  help="The share of activity brought to the UK by the credit at which the tax it brings in equals "
                       "its cost, at the displacement chosen.")
        grid = pd.DataFrame({"Share brought to the UK": np.linspace(0, 1, 101)})
        grid["Gain or loss, £m"] = [net_fiscal_impact(T["receipts_elig"], T["net_cost"], a, disp)["Net fiscal impact"]
                                    for a in grid["Share brought to the UK"]]
        line = alt.Chart(grid).mark_line(color=C1, strokeWidth=2).encode(
            x=alt.X("Share brought to the UK:Q", axis=_axis("Share of activity brought to the UK by the credit",
                                                           fmt="%", values=[i / 10 for i in range(11)])),
            y=alt.Y("Gain or loss, £m:Q", axis=_axis("Gain or loss to the Exchequer, £m")),
            tooltip=[alt.Tooltip("Share brought to the UK:Q", format=".0%"),
                     alt.Tooltip("Gain or loss, £m:Q", format=",.1f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
        point = alt.Chart(pd.DataFrame({"Share brought to the UK": [add], "Gain or loss, £m": [nf["Net fiscal impact"]]})
                          ).mark_point(filled=True, size=110, color=C1, stroke=SURFACE, strokeWidth=2).encode(
            x="Share brought to the UK:Q", y="Gain or loss, £m:Q")
        st.altair_chart((line + zero + point).properties(height=280).configure_view(stroke=None), width="stretch")

        st.subheader("Credit and tax by production")
        tbl = summ.loc[[p.production for p in sel], ["format", "uk_core_expenditure", "qualifying_within_cap",
                                                     "uk_vfx", "gross_credit", "net_cost_of_relief",
                                                     "gross_receipts", "gross_receipts_per_1_relief",
                                                     "break_even_additionality"]]
        tbl.index = [short(i) for i in tbl.index]
        tbl.index.name = "Production"
        tbl.columns = ["Format", "UK spend", "Within cap", "UK VFX", "Credit", "After tax", "Tax paid",
                       "Tax per £1", "Pays above"]
        tbl.index = [f"{i} ({FORMAT_NAMES.get(f, f)})" for i, f in zip(tbl.index, tbl["Format"])]
        tbl.index.name = "Production"
        tbl = tbl.drop(columns="Format").astype(float)
        tbl["Tax per £1"] = tbl["Tax per £1"].map(lambda v: f"£{v:.2f}" if pd.notna(v) else "not eligible")
        tbl["Pays above"] = tbl["Pays above"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "not eligible")
        st.dataframe(tbl.style.format({c: "{:,.1f}" for c in tbl.columns[:6]}), width="stretch")
        st.caption("£m. UK spend: UK production spending that counts towards the credit. Within cap: the part "
                   "below 80% of total production spending. UK VFX: UK visual effects spending. After tax: the "
                   "credit's cost to the Exchequer. Pays above: the share of activity the credit must bring to the "
                   "UK for the tax it brings in to cover its cost, with no displacement. Assumes every eligible "
                   "production is certified as British and claims in a single year. Tax uses average rates.")

# --------------------------------------------------------------------------------------
# Tab 3: response to price (econometrics)
# --------------------------------------------------------------------------------------
with tab_price:
    headline_price = st.container()
    st.markdown(
        "A service that has raised prices in some markets and not others can measure how many members a price rise "
        "costs. This view compares, month by month, markets that raised their price with markets that did not, "
        "using membership data the service already holds. The estimate feeds the levy calculation in the next tab.")
    c1, c2 = st.columns([1, 1])
    panel_src = c1.radio("Membership data", ["Illustrative markets", "Upload membership data"], horizontal=True)
    panel = price_panel()
    if panel_src == "Upload membership data":
        up_panel = c1.file_uploader("Monthly price and members by market (CSV)", type="csv")
        if up_panel is not None:
            panel = pd.read_csv(up_panel, comment="#")
        else:
            c1.info("Upload a file to replace the illustrative markets.")
    c2.download_button("Download the membership data template", price_panel().to_csv(index=False),
                       "membership_template.csv", "text/csv")
    try:
        est = estimate_response(panel)
    except (ValueError, KeyError, np.linalg.LinAlgError) as err:
        st.error(f"The estimate could not be made from this data.\n\n{err}")
        est = None

    if est is not None:
        illustrative = panel_src == "Illustrative markets" or panel.equals(price_panel())
        check = (f" The illustrative data were built with a true value of 1.2, which lies within the likely range."
                 if illustrative and est.low <= 1.2 <= est.high else "")
        with headline_price:
            in_short(f"Comparing {est.n_rising} markets that raised their price with {est.n_comparison} that did not, "
                     f"a 1% price rise loses about {est.long_run:.2f}% of members within a year (likely range "
                     f"{est.low:.2f}% to {est.high:.2f}%). A simple before-and-after comparison suggests only "
                     f"{est.before_after:.2f}%, because membership was growing in every market at the same time.{check}")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Members lost per 1% rise", f"{est.long_run:.2f}%", border=True,
                  help="Average response 9 to 12 months after the rise: the long-run price elasticity of membership.")
        e2.metric("Likely range", f"{est.low:.2f}–{est.high:.2f}%", border=True,
                  help="95% confidence interval, allowing for markets that appear in more than one comparison.")
        e3.metric("Simple before-and-after", f"{est.before_after:.2f}%", border=True,
                  help="Change in members in markets that raised prices, six months before against months 6 to 11 "
                       "after, with no comparison group.")
        e4.metric("Markets compared", f"{est.n_rising} and {est.n_comparison}", border=True,
                  help="Markets that raised their price, and markets that never changed price in the data. Markets "
                       "that raised prices later also act as comparisons until they do.")

        st.subheader("Members lost per 1% price rise, by months since the rise")
        path = est.path.rename(columns={"months_since_rise": "Months since the rise", "elasticity": "Members lost, %"})
        band = alt.Chart(path).mark_area(color=C1, opacity=0.18).encode(
            x=alt.X("Months since the rise:Q", axis=_axis("Months since the price rise",
                                                         values=list(range(-6, 13)))),
            y=alt.Y("low:Q", axis=_axis("Members lost per 1% price rise, %")), y2="high:Q")
        line = alt.Chart(path).mark_line(color=C1, strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True)
                                         ).encode(
            x="Months since the rise:Q", y="Members lost, %:Q",
            tooltip=[alt.Tooltip("Months since the rise:Q"), alt.Tooltip("Members lost, %:Q", format=".2f"),
                     alt.Tooltip("low:Q", title="Range from", format=".2f"),
                     alt.Tooltip("high:Q", title="Range to", format=".2f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
        rise_rule = alt.Chart(pd.DataFrame({"x": [-0.5]})).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(x="x:Q")
        rise_lab = alt.Chart(pd.DataFrame({"x": [-0.5], "t": ["price rise"]})).mark_text(
            align="left", dx=5, color=INK2, fontSize=11).encode(x="x:Q", y=alt.value(12), text="t:N")
        st.altair_chart((band + zero + rise_rule + rise_lab + line).properties(height=300).configure_view(stroke=None),
                        width="stretch")
        st.caption("Before the rise the two groups move together, so the markets that did not raise prices are a fair "
                   "guide to what would have happened without the rise. After it, members leave over the first few "
                   "months as cancellations take effect. The shaded band is the likely range.")
        with st.expander("Method"):
            st.markdown(
                "Each month in which a price rise took place is compared separately. The markets that raised prices "
                "that month are set against markets that did not change price over the 6 months before and 12 months "
                "after, including markets whose own rise came later. Each comparison removes differences in size "
                "between markets and changes common to all markets in a given month, and the size of each rise scales "
                "the response, so every point is a percentage loss of members per 1% price rise. This is a stacked "
                "difference-in-differences event study. Comparing each rise only with markets that have not yet "
                "changed price avoids the bias a single regression across all markets can show when rises happen at "
                "different times and the response builds up over months. The likely range allows for the same market "
                "appearing in several comparisons (standard errors clustered by market). The estimate assumes no other "
                "change hit the rising markets at the moment of their rise; promotions or new content timed to a "
                "price rise would need to be added to the comparison.")


# --------------------------------------------------------------------------------------
# Tab 3: subscriptions and levies
# --------------------------------------------------------------------------------------
with tab_subs:
    headline_subs = st.container()
    mem0 = json.loads((EXAMPLES / "illustrative_membership.json").read_text(encoding="utf-8"))
    st.markdown(
        "A subscription service already holds what this view needs: its members, what they pay, and how sign-ups "
        "and cancellations responded to past price changes. The view estimates how much members value the "
        "service above what they pay, and who would bear a levy on subscription revenue. Membership is assumed to "
        "fall in a straight line as price rises, which gives a cautious estimate of the value to members.")
    s1, s2, s3 = st.columns(3)
    members = s1.number_input("Members (millions)", 0.1, 100.0, float(mem0["members_m"]), 0.5)
    price = s2.number_input("Average monthly revenue per member (£)", 1.0, 50.0, float(mem0["monthly_price"]), 0.5)
    est_ok = est is not None and est.long_run > 0
    opts = ([f"Estimated from price rises ({est.long_run:.2f})"] if est_ok else []) + ["Enter a figure"]
    el_mode = s3.radio("How strongly membership responds to price", opts,
                       help="Price elasticity: 1.2 means a 1% price rise loses 1.2% of members. The estimate comes "
                            "from the Response to price tab.")
    elast = est.long_run if el_mode.startswith("Estimated") else s3.number_input(
        "Members lost per 1% price rise", 0.1, 5.0, 1.2, 0.1)
    s4, s5, s6 = st.columns(3)
    levy_rate = s4.slider("Levy on subscription revenue", 0.0, 10.0, 100 * float(mem0["levy_rate"]), 0.5,
                          format="%.1f%%") / 100
    mc_mode = s5.radio("Cost of serving one more member", ["Implied by the current price", "Enter a figure"],
                       help="Implied: the cost at which the current price is the one that earns the most profit.")
    mc_share = None
    if mc_mode == "Enter a figure":
        mc_share = s5.slider("As a share of price", 0, 90, 10, 5, format="%d%%") / 100
    try:
        mem = Membership(members, price, elast, mc_share)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    pt_opt = profit_max_pass_through(mem, levy_rate)
    pt_mode = s6.radio("Share of the levy passed on in prices", ["What a profit-maximising service would choose",
                                                                 "Enter a figure"])
    pt = pt_opt if pt_mode.startswith("What") else s6.slider("Share passed on", 0, 100, 100, 5, format="%d%%") / 100
    li = levy_incidence(mem, levy_rate, pt)
    lo = levy_incidence(mem, levy_rate, pt_opt)

    with headline_subs:
        in_short(f"With a 1% price rise losing {elast:.2f}% of members, a {100 * levy_rate:g}% levy on subscription "
                 f"revenue would raise about {gbp(lo['levy_raised_m'], 0)} a "
                 f"year. A service that sets its price to earn the most profit would pass on about {pt_opt:.0%} of "
                 f"it, so members would bear {gbp(lo['borne_by_members_m'], 0)} and the service "
                 f"{gbp(lo['borne_by_service_m'], 0)}. Members value the service at about "
                 f"{gbp(mem.consumer_surplus_m(), 0)} a year above what they pay.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Value to members", gbp(mem.consumer_surplus_m(), 0), border=True,
              help="Consumer surplus: what members would be willing to pay above what they pay, per year.")
    k2.metric("Levy raised a year", gbp(li["levy_raised_m"]), border=True)
    k3.metric("Cost to members", gbp(li["borne_by_members_m"]), border=True,
              help=f"Price rises {pt * levy_rate:.2%}; about {li['members_lost_m'] * 1000:,.0f} thousand members leave.")
    k4.metric("Cost to the service", gbp(li["borne_by_service_m"]), border=True)
    st.caption(f"Cost of serving one more member: {mem.marginal_cost_share:.0%} of price. Share of the levy passed "
               f"on: {pt:.0%}. Cost to members and the service beyond the levy raised: "
               f"{gbp(li['excess_burden_m'])} a year.")

    curve = []
    for x in np.linspace(0, 1, 41):
        r = levy_incidence(mem, levy_rate, x)
        curve += [{"Passed on": x, "Series": "Cost to members", "£m": r["borne_by_members_m"]},
                  {"Passed on": x, "Series": "Cost to the service", "£m": r["borne_by_service_m"]},
                  {"Passed on": x, "Series": "Total cost", "£m": r["borne_by_members_m"] + r["borne_by_service_m"]},
                  {"Passed on": x, "Series": "Levy raised", "£m": r["levy_raised_m"]}]
    cd = pd.DataFrame(curve)
    series = ["Cost to members", "Cost to the service", "Total cost", "Levy raised"]
    lines = alt.Chart(cd).mark_line(strokeWidth=2).encode(
        x=alt.X("Passed on:Q", axis=_axis("Share of the levy passed on in prices", fmt="%",
                                          values=[i / 10 for i in range(11)])),
        y=alt.Y("£m:Q", axis=_axis("£m a year")),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=[C1, C2, C3, INK2]), legend=_legend()),
        strokeDash=alt.condition(alt.datum.Series == "Levy raised", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("Passed on:Q", format=".0%"), "Series:N", alt.Tooltip("£m:Q", format=",.1f")])
    rule = alt.Chart(pd.DataFrame({"x": [pt_opt]})).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(x="x:Q")
    rule_lab = alt.Chart(pd.DataFrame({"x": [pt_opt], "t": ["profit-maximising"]})).mark_text(
        align="left", dx=5, dy=-6, color=INK2, fontSize=11).encode(x="x:Q", y=alt.value(12), text="t:N")
    st.altair_chart((lines + rule + rule_lab).properties(height=300).configure_view(stroke=None), width="stretch")
    st.caption("A levy on revenue takes the same share of revenue at every price, so it gives a service with low "
               "running costs little reason to raise its price. Passing on more than that loses members who paid "
               "more than they cost to serve. The response to price comes from the Response to price tab.")

# --------------------------------------------------------------------------------------
# Tab 4: how it works
# --------------------------------------------------------------------------------------
with tab_method:
    st.markdown("""
**From spending to the wider economy.** Each line of a production's cost report is matched to one of 104 industries
in the official input-output tables for the UK (ONS, 2023). These tables record what every industry buys from every
other, so the model can follow each pound through suppliers and their suppliers. Money spent on imports or paid in
tax on purchases is taken out first, because it does not reach UK businesses. Cast and crew pay counts directly.
Counting the spending of wages adds the activity supported in shops, restaurants and other services.

**Checked against official figures.** Run with the official average for each industry, the model reproduces the ONS
published figures for all 104 industries. The checks, and automated tests that run on every change to the code, are
in the source code.

**Tax credit and tax paid.** The credit follows the published rules for the Audio-Visual Expenditure Credit,
including the 80% cap and the separate treatment of visual effects. Tax paid applies average tax rates to the
wages, spending and profits supported. Because these figures count all the activity, whether the credit pays for
itself depends on the share of activity it brought to the UK.

**How members respond to price.** The response is estimated from markets that raised their price, compared month by
month with markets that did not (a stacked difference-in-differences event study, with standard errors clustered by
market). A simple before-and-after comparison would mistake growth common to all markets for a weak response to price.

**Members and levies.** Membership falls in a straight line as price rises, at the estimated rate, which gives a
cautious estimate of the value to members. Unless a figure is entered, the cost of serving one more member is set so
that the current price is the one that earns the most profit.

**Limits.** The model assumes businesses buy their inputs in fixed proportions and have spare capacity. It shows the
activity linked to the spending at today's structure of the economy, and would overstate the knock-on effects when
workers or suppliers are scarce. The official tables cover the UK as a whole, so regional figures are limited to
spending, pay and reported jobs. Tax figures use average rates, while many crew are freelance. Effects such as
screen tourism, skills and exports are not included.

**Using it in other markets.** Everything specific to the UK sits in three places: the national input-output tables,
a file of tax credit and tax rules, and the household spending assumptions. With each country's tables and rules,
checked against its statistics office's published figures, the same cost-report mapping would give comparable
results for every market where productions are made.

**Research that would firm up the figures.** Four pieces of work would strengthen the evidence: estimates of how
location decisions respond to the credit, which set the share of activity it brings to the UK; a fuller model of how
members respond to price, adding promotions, new releases and competitors' prices, and separating sign-ups from
cancellations; regional input-output tables, for effects by nation and region; and measures of screen tourism and
exports, which the model leaves out.
""")
    if url:
        st.markdown(f"Full method, sources and code: [{url}]({url}). One-page technical and policy note: "
                    f"[technical_note.pdf]({url}/blob/main/docs/technical_note.pdf).")
