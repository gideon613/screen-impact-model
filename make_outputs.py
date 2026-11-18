"""
Build the exemplar output pack from the illustrative inputs in examples/.

    python make_outputs.py

Writes to outputs/:
    impact_summary.md          plain-English summary with the headline figures
    slate_summary.csv          one row per production plus the slate total
    spend_bridge.csv           budget to first-round UK demand, by production
    exchequer.csv              relief and gross receipts by tax, by production
    net_fiscal.csv             net fiscal impact of the credit by additionality and displacement
    region.csv                 UK spend, own cast and crew pay and reported direct jobs by nation and
                               region (spending and direct figures; no regional multipliers)
    industry.csv               GVA and jobs by industry for the whole slate
    sensitivity.csv            gross receipts per £1 of relief under alternative settings
    levy_incidence.csv         consumer surplus and levy incidence for a subscription service
    validation.csv             the model's reproduction of ONS results and its accounting identities
    *.png                      charts used in the README
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from fiscal import break_even_additionality, load_config, net_fiscal_impact
from io_footprint import FootprintModel, load_data
from slate import default_settings, run_slate, summary_table
from welfare import Membership, levy_incidence, profit_max_pass_through

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SLATE = ROOT / "examples" / "illustrative_slate.csv"
PAYROLL = ROOT / "examples" / "illustrative_payroll.csv"
DISPLACEMENT_CASES = [0.0, 0.25]   # illustrative displacement shares for the net fiscal chart

# Chart tokens: validated categorical slots on the light surface
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, UP, DOWN = "#898781", "#2a78d6", "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def stacked_barh(ax, labels, parts: dict, colours: dict, fmt="{:.1f}", min_label=None, totals=True):
    left = [0.0] * len(labels)
    span = max(sum(v[i] for v in parts.values()) for i in range(len(labels)))
    min_label = min_label if min_label is not None else span * 0.06
    for name, values in parts.items():
        bars = ax.barh(labels, values, left=left, height=0.55, color=colours[name], edgecolor=SURFACE,
                       linewidth=2, label=name)
        for b, v in zip(bars, values):
            if v >= min_label:
                ink = INK if colours[name] == C4 else "white"
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, fmt.format(v),
                        ha="center", va="center", color=ink, fontsize=8.5, fontweight="bold")
        left = [l + v for l, v in zip(left, values)]
    if totals:
        for y, total in enumerate(left):
            ax.text(total + span * 0.012, y, fmt.format(total), va="center", color=INK, fontsize=9,
                    fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.14)
    style(ax)


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)


def eligible_ratio(prods):
    rec = sum(p.receipts["Total receipts"] for p in prods if p.relief.eligible)
    cost = sum(p.relief.net_cost for p in prods if p.relief.eligible)
    return rec / cost


def main():
    data = load_data()
    cfg = load_config("uk")
    central = default_settings(cfg)
    prods, lines, ctx = run_slate(SLATE, central, data=data, cfg=cfg, payroll=PAYROLL)
    model = ctx["model"]

    # ---------------- tables ----------------
    summ = summary_table(prods, model)
    summ.round(3).to_csv(OUT / "slate_summary.csv")

    bridge = pd.DataFrame({p.production: p.bridge for p in prods})
    bridge["Slate total"] = bridge.sum(axis=1)
    bridge.round(3).to_csv(OUT / "spend_bridge.csv")

    ex_rows = {}
    for p in prods:
        r = {k: v for k, v in p.receipts.items() if not k.startswith("_")}
        r.update({"AVEC gross credit": p.relief.gross_credit, "AVEC net cost": p.relief.net_cost})
        ex_rows[p.production] = r
    exch = pd.DataFrame(ex_rows)
    exch["Slate total"] = exch.sum(axis=1)
    exch.round(3).to_csv(OUT / "exchequer.csv")

    elig = [p for p in prods if p.relief.eligible]
    rec_e = sum(p.receipts["Total receipts"] for p in elig)
    cost_e = sum(p.relief.net_cost for p in elig)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    nf = pd.DataFrame([{"additionality": a, "displacement": d,
                        **net_fiscal_impact(rec_e, cost_e, a, d)} for d in DISPLACEMENT_CASES for a in grid])
    nf.round(3).to_csv(OUT / "net_fiscal.csv", index=False)
    breakeven = {d: break_even_additionality(rec_e, cost_e, d) for d in DISPLACEMENT_CASES}

    reg = pd.concat([p.by_region for p in prods]).groupby(level=0).sum()
    reg["share_of_uk_spend"] = reg["uk_spend"] / reg["uk_spend"].sum()
    reg = reg.sort_values("uk_spend", ascending=False)
    reg.round(3).to_csv(OUT / "region.csv")

    ind = sum(p.results.table.drop(columns="industry") for p in prods)
    ind.insert(0, "industry", data.names)
    ind["gva_total"] = ind[["gva_direct", "gva_indirect", "gva_induced"]].sum(axis=1)
    ind["fte_total"] = ind[["fte_direct", "fte_indirect", "fte_induced"]].sum(axis=1)
    ind = ind.sort_values("gva_total", ascending=False)
    ind[["industry", "gva_direct", "gva_indirect", "gva_induced", "gva_total", "fte_total"]].round(3).to_csv(
        OUT / "industry.csv")

    val = FootprintModel(central, data).validate()
    val.to_csv(OUT / "validation.csv", index=False)

    # ---------------- sensitivity (gross receipts per £1 of relief, eligible productions) ----------------
    base = eligible_ratio(prods)

    def ratio(settings=None, cfg_edit=None):
        c = copy.deepcopy(cfg)
        if cfg_edit:
            cfg_edit(c)
        ps, _, _ = run_slate(SLATE, settings or central, data=data, cfg=c, payroll=PAYROLL)
        return eligible_ratio(ps)

    def no_ct(c):
        c["exchequer"]["corporation_tax_receipts_gbp_m"] = 0.0

    def wedge_2024(c):
        c["exchequer"]["labour_tax_rate_on_compensation"] = 0.294

    tests = [
        ("Induced effects", ratio(replace(central, include_induced=False)), base,
         "excluded (Type I)", "included"),
        ("Import leakage", ratio(replace(central, import_uplift=0.25)), ratio(replace(central, remove_imports=False)),
         "import shares +25%", "no import adjustment"),
        ("Corporation tax", ratio(cfg_edit=no_ct), base, "excluded", "included"),
        ("Tax on labour", ratio(cfg_edit=wedge_2024), base, "2024 tax wedge, 29.4%", "2023, 31.2%"),
        ("Household closure", ratio(replace(central, saving_ratio=0.101)), ratio(replace(central, tax_wedge=0.294)),
         "saving ratio 10.1% (2024)", "tax wedge 29.4% (2024)"),
    ]
    sens = []
    for name, a, b, la, lb in tests:
        lo, hi = (a, b) if a <= b else (b, a)
        llo, lhi = (la, lb) if a <= b else (lb, la)
        sens.append({"setting": name, "low": lo, "high": hi, "low_case": llo, "high_case": lhi, "range": hi - lo})
    sens = pd.DataFrame(sens).sort_values("range", ascending=False)
    sens.round(3).to_csv(OUT / "sensitivity.csv", index=False)

    # ---------------- welfare ----------------
    w = json.loads((ROOT / "examples" / "illustrative_membership.json").read_text(encoding="utf-8"))
    mem = Membership(w["members_m"], w["monthly_price"], w["elasticity"], w["marginal_cost_share"])
    pt_opt = profit_max_pass_through(mem, w["levy_rate"])
    cases = sorted(set(w["pass_through_cases"]) | {round(pt_opt, 4)})
    levy = pd.DataFrame([levy_incidence(mem, w["levy_rate"], pt) for pt in cases])
    levy["profit_maximising"] = np.isclose(levy["pass_through"], round(pt_opt, 4))
    levy.round(3).to_csv(OUT / "levy_incidence.csv", index=False)

    # ---------------- charts ----------------
    short = {p.production: p.production.split(" (")[0] for p in prods}
    eff_col = {"Direct": C1, "Indirect": C2, "Induced": C3}

    # 1. GVA by production and effect
    sub = summ.drop(index="Slate total")
    labels = [f"{short[i]}\n{sub.loc[i, 'uk_spend']:.1f}m UK spend" for i in sub.index]
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=200)
    stacked_barh(ax, labels, {"Direct": sub["gva_direct"].tolist(), "Indirect": sub["gva_indirect"].tolist(),
                              "Induced": sub["gva_induced"].tolist()}, eff_col)
    ax.set_xlabel("UK gross value added supported, £m (illustrative inputs in 2023 prices)")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    title(fig, "GVA supported by each production")
    fig.tight_layout(); fig.savefig(OUT / "gva_by_production.png"); plt.close(fig)

    # 2. Who receives the GVA: wages, operating surplus, taxes on production, by effect
    tot = sum(p.results.summary() for p in prods)
    inc_col = {"Wages": C1, "Operating surplus": C2, "Taxes on production": C4}
    eff = ["direct", "indirect", "induced"]
    fig, ax = plt.subplots(figsize=(8.4, 2.6), dpi=200)
    stacked_barh(ax, ["Direct", "Indirect (supply chain)", "Induced (household spending)"],
                 {"Wages": [tot.loc["coe (£m)", k] for k in eff],
                  "Operating surplus": [tot.loc["gos (£m)", k] for k in eff],
                  "Taxes on production": [max(tot.loc["ptax (£m)", k], 0) for k in eff]}, inc_col)
    ax.set_xlabel("Slate GVA by income, £m")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    title(fig, "Who receives the GVA: workers, businesses and government")
    fig.tight_layout(); fig.savefig(OUT / "gva_by_income.png"); plt.close(fig)

    # 3. Exchequer: net cost of relief against gross receipts, eligible productions
    tax_col = {"Income tax and NICs": C1, "VAT and duties on household spending": C2,
               "Corporation tax": C3, "Taxes on production": C4}
    fig, ax = plt.subplots(figsize=(8.4, 3.3), dpi=200)
    labs = [short[p.production] for p in elig]
    y = list(range(len(labs)))
    h = 0.36
    cost = [p.relief.net_cost for p in elig]
    ax.barh([v - h / 2 - 0.02 for v in y], cost, height=h, color=NEUTRAL, edgecolor=SURFACE, linewidth=2,
            label="Net cost of AVEC")
    left = [0.0] * len(elig)
    for name, col in tax_col.items():
        vals = [p.receipts[name] + (p.receipts["Product taxes on purchases"] if name == "Taxes on production" else 0)
                for p in elig]
        ax.barh([v + h / 2 + 0.02 for v in y], vals, left=left, height=h, color=col, edgecolor=SURFACE,
                linewidth=2, label=name if name != "Taxes on production" else "Taxes on production and purchases")
        left = [l + v for l, v in zip(left, vals)]
    for i, p in enumerate(elig):
        ax.text(cost[i] + 0.3, i - h / 2 - 0.02, f"{cost[i]:.1f}", va="center", fontsize=8.5, color=INK2)
        ax.text(left[i] + 0.3, i + h / 2 + 0.02, f"{left[i]:.1f}  (£{left[i] / cost[i]:.2f} per £1, gross)",
                va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(y, labs)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.55)
    ax.set_xlabel("£m")
    style(ax)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    title(fig, "Gross tax receipts supported against the net cost of the credit")
    fig.tight_layout(); fig.savefig(OUT / "exchequer.png"); plt.close(fig)

    # 4. Net fiscal impact against additionality
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=200)
    line_col = {DISPLACEMENT_CASES[0]: C1, DISPLACEMENT_CASES[1]: C2}
    for d in DISPLACEMENT_CASES:
        sub_nf = nf[nf["displacement"] == d]
        ax.plot(sub_nf["additionality"] * 100, sub_nf["Net fiscal impact"], color=line_col[d], linewidth=2,
                label=f"Displacement {d:.0%}")
        end = sub_nf.iloc[-1]
        ax.text(101.5, end["Net fiscal impact"], f"{d:.0%} displacement", va="center", fontsize=8.5, color=INK2)
        be = breakeven[d]
        if be <= 1:
            ax.plot([be * 100], [0], marker="o", markersize=7, color=line_col[d], markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=5)
            first = d == DISPLACEMENT_CASES[0]
            ax.annotate(f"break-even {be:.0%}", (be * 100, 0), xytext=(-6, 10) if first else (6, -16),
                        textcoords="offset points", ha="right" if first else "left", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xlim(0, 118)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("Additionality: share of the activity that would not have taken place in the UK without the credit")
    ax.set_ylabel("£m", color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    title(fig, "Net fiscal impact of the credit depends on additionality and displacement")
    fig.tight_layout(); fig.savefig(OUT / "net_fiscal.png"); plt.close(fig)

    # 5. Tornado: gross receipts per £1 of relief
    fig, ax = plt.subplots(figsize=(8.4, 3.1), dpi=200)
    for yy, (_, r) in enumerate(sens.iterrows()):
        if r["low"] < base - 1e-9:
            ax.barh(yy, r["low"] - base, left=base, height=0.5, color=DOWN, edgecolor=SURFACE, linewidth=2)
            ax.text(r["low"] - 0.01, yy, f"{r['low']:.2f}  {r['low_case']}", ha="right", va="center",
                    fontsize=8, color=INK2)
        if r["high"] > base + 1e-9:
            ax.barh(yy, r["high"] - base, left=base, height=0.5, color=UP, edgecolor=SURFACE, linewidth=2)
            ax.text(r["high"] + 0.01, yy, f"{r['high']:.2f}  {r['high_case']}", ha="left", va="center",
                    fontsize=8, color=INK2)
    ax.axvline(base, color=INK, linewidth=1)
    ax.set_yticks(range(len(sens)), sens["setting"].tolist())
    ax.invert_yaxis()
    span = max(base - sens["low"].min(), sens["high"].max() - base)
    ax.set_xlim(base - span * 2.4, base + span * 2.4)
    ax.set_xlabel(f"Gross receipts per £1 of net relief (central case £{base:.2f})")
    style(ax)
    title(fig, "Sensitivity of the gross Exchequer return to model settings")
    fig.tight_layout(); fig.savefig(OUT / "sensitivity_tornado.png"); plt.close(fig)

    # 6. Levy incidence
    lev_col = {"Borne by members": C1, "Borne by the service": C2}
    labs = [f"{pt:.0%} passed on" + ("\n(profit-maximising)" if opt else "")
            for pt, opt in zip(levy["pass_through"], levy["profit_maximising"])]
    fig, ax = plt.subplots(figsize=(8.4, 3.2), dpi=200)
    stacked_barh(ax, labs, {"Borne by members": levy["borne_by_members_m"].tolist(),
                            "Borne by the service": levy["borne_by_service_m"].tolist()}, lev_col, fmt="{:.0f}")
    for i, r in levy.iterrows():
        ax.plot([r["levy_raised_m"]], [i - 0.4], marker="v", markersize=7, color=INK, linestyle="none")
    ax.plot([], [], marker="v", markersize=7, color=INK, linestyle="none", label="Levy raised")
    ax.set_xlabel(f"£m a year: {w['levy_rate']:.0%} levy on subscription revenue, hypothetical service")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    title(fig, "Who bears a levy on subscription revenue")
    fig.tight_layout(); fig.savefig(OUT / "levy_incidence.png"); plt.close(fig)

    # ---------------- plain-English summary ----------------
    T = summ.loc["Slate total"]
    elig_names = [short[p.production].lower() for p in elig]
    london = reg.loc["London", "share_of_uk_spend"] if "London" in reg.index else 0.0
    nonlon = reg.drop(index="London", errors="ignore")
    the = lambda r: f"the {r}" if r in ("South East", "North West", "North East", "East Midlands",
                                          "West Midlands", "East of England", "South West") else r
    top_two = " and ".join(the(r) for r in nonlon.index[:2])
    ineligible = [short[p.production].lower() for p in prods if not p.relief.eligible]
    cs = mem.consumer_surplus_m()
    lopt = levy[levy["profit_maximising"]].iloc[0]
    l100 = levy[levy["pass_through"] == 1.0].iloc[0]
    nf50 = net_fiscal_impact(rec_e, cost_e, 0.5, 0.0)["Net fiscal impact"]
    gross_ratio = rec_e / cost_e
    fte_note = ("as reported in payroll" if T["direct_fte_source"] == "reported"
                else "as modelled from their pay")
    text = f"""# Economic contribution of an illustrative UK production slate

All inputs are illustrative. They show the outputs the model produces from production cost reports,
payroll and membership data. They do not describe any real production, studio, service or company.
Money figures are in the prices of the inputs, which are stated in 2023 prices.

## Headline

The four productions had combined budgets of £{T['budget']:.1f}m, of which £{T['uk_spend']:.1f}m was spent
in the UK. Including their UK supply chains and the spending of the wages they paid, that spend
supported £{T['gva_total']:.1f}m of UK gross value added and {T['fte_total']:,.0f} full-time equivalent jobs,
{T['fte_direct']:,.0f} of them on the productions themselves {fte_note}. Of the GVA, £{T['wages']:.1f}m was
paid to workers as wages, £{T['operating_surplus']:.1f}m was operating surplus earned by UK businesses and
the rest was taxes on production.

## Where the money was spent

{1 - london:.0%} of UK spend took place outside London, led by {top_two}. These figures show where the
productions spent money, paid cast and crew and reported jobs. The ONS tables describe the UK as a
whole, so the model does not estimate supply-chain or household-spending effects for individual
nations and regions.

## Tax credit and the public finances

The {", ".join(elig_names[:-1])} and {elig_names[-1]} would qualify for the Audio-Visual Expenditure Credit; the
{" and ".join(ineligible)} does not, as unscripted entertainment is outside the scheme. The three
eligible productions would claim an estimated £{exch.loc['AVEC gross credit', 'Slate total']:.1f}m of credit,
a net cost to the Exchequer of £{cost_e:.1f}m once the credit is taxed. The activity they support
generates £{rec_e:.1f}m of tax receipts, mostly income tax and National Insurance on the wages paid, or
£{gross_ratio:.2f} of gross receipts for every £1 of net relief.

These receipts are gross. They count tax on all the activity the productions support, including any
that would have taken place in the UK without the credit, so they measure contribution. The effect of
the credit on the public finances depends on two further shares: additionality, the share of the
activity that would not have taken place in the UK without the credit, and displacement, the share
of that additional activity that draws on workers and resources that would otherwise have been
employed elsewhere in the UK. With no displacement, the receipts caused by the credit cover its cost
once additionality reaches {breakeven[0.0]:.0%}. With {DISPLACEMENT_CASES[1]:.0%} displacement the threshold rises to
{breakeven[DISPLACEMENT_CASES[1]]:.0%}. At 50% additionality and no displacement, the credit would cost the
Exchequer £{-nf50:.1f}m net. Across the model settings tested, the gross return ranges from
£{sens['low'].min():.2f} to £{sens['high'].max():.2f} per £1, and the widest range comes from
{sens.iloc[0]['setting'].lower()}.

## Value to members and the incidence of a levy

For a hypothetical service with {w['members_m']:.0f} million members paying £{w['monthly_price']:.2f} a month and a
membership price elasticity of {w['elasticity']}, members receive about £{cs:,.0f}m a year of value above
what they pay (linear demand, the conservative case). A {w['levy_rate']:.0%} levy on subscription revenue
would raise about £{lopt['levy_raised_m']:.0f}m a year. A service that sets its price to maximise profit, with a
cost of serving each member consistent with that elasticity, would pass on about {lopt['pass_through']:.0%} of
the levy. Its price would rise by about {(lopt['new_price'] - w['monthly_price']) * 100:.0f}p a month, around
{lopt['members_lost_m'] * 1000:,.0f} thousand members would leave, members would lose £{lopt['borne_by_members_m']:.0f}m of value
and the service would bear £{lopt['borne_by_service_m']:.0f}m. A levy on revenue takes the same share of revenue
at every price, so it gives a service with low running costs little reason to raise its price. If
the service passed the levy on in full, about {l100['members_lost_m'] * 1000:,.0f} thousand members would leave and
the combined loss to members and the service would exceed the levy raised by
£{l100['excess_burden_m']:.0f}m.

## Basis

Static input-output model of the UK economy built from the ONS Input-Output Analytical Tables 2023,
reproducing the ONS published Type I effects. Induced effects use a household closure based on the
OECD tax wedge and ONS saving ratio for 2023. Tax receipts use average effective rates and are gross.
Credit calculations follow the AVEC rules in a single-period simplification and assume every eligible
production is certified as British.
"""
    (OUT / "impact_summary.md").write_text(text, encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
    print(summ.T)
    print(sens.to_string(index=False))
    print(levy.T)
    print(reg)
    print(text)


if __name__ == "__main__":
    main()
