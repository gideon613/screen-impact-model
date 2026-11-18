"""Tests for the price-response estimates. Run with: python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elasticity import PANEL, check_panel, event_study, simulate_panel  # noqa: E402


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_event_study_recovers_the_true_elasticity(seed):
    r = event_study(simulate_panel(seed=seed, elasticity=1.2))
    assert abs(r.long_run - 1.2) < 3 * r.long_run_se + 0.03
    pre = r.path[r.path["months_since_rise"] < -1]
    assert (pre["elasticity"].abs() < 0.08).all()          # no response before the rise


def test_before_and_after_is_biased_by_common_growth():
    r = event_study(simulate_panel(elasticity=1.2))
    assert r.before_after < 0.6 < r.long_run


def test_no_price_rises_means_no_response():
    r = event_study(simulate_panel(elasticity=0.0))
    assert abs(r.long_run) < 3 * r.long_run_se + 0.03


def test_illustrative_file_matches_the_simulation():
    saved = pd.read_csv(PANEL, comment="#")
    pd.testing.assert_frame_equal(saved, simulate_panel(), check_dtype=False)


def test_panel_problems_are_reported():
    p = simulate_panel()
    m = p["market"] == "Market 01"
    p.loc[m & (p["month"] == "2023-06"), "price"] = 99.0            # a rise and then a fall
    p = p[~((p["market"] == "Market 02") & (p["month"] == "2022-03"))]  # a missing month
    errors = " ".join(check_panel(p))
    assert "Price falls" in errors and "balanced panel" in errors
    assert check_panel(p.drop(columns="price")) == [
        "Membership data is missing columns: ['price']. Expected ['market', 'month', 'price', 'members_k']"]
