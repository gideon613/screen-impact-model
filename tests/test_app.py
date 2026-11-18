"""Runs the Streamlit demonstrator headless and checks that it renders and responds to its controls
without errors. Skipped if Streamlit is not installed. Run with: python -m pytest -q"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_headline_figures_match_the_engine(app):
    assert [t.label for t in app.tabs] == ["Economic contribution", "Tax credit", "Response to price",
                                           "Subscriptions and levies", "How it works"]
    assert _metric(app, "Members lost per 1% rise") == "1.16%"
    assert _metric(app, "Value to members") == "£518m"
    assert _metric(app, "UK spend") == "£143.3m"
    assert _metric(app, "UK GDP supported") == "£156.3m"
    assert _metric(app, "Tax paid per £1 the credit costs") == "£1.50"
    assert _metric(app, "Credit pays for itself above") == "67%"


def test_controls_rerun_without_errors(app):
    labels = [s.label for s in app.slider]
    app.slider[labels.index("Bring spending now made abroad into the UK")].set_value(100).run()
    assert not app.exception
    assert _metric(app, "UK spend") == "£152.8m"
    app.selectbox[0].set_value("Feature film").run()
    radios = [r.label for r in app.radio]
    app.radio[radios.index("Share of the levy passed on in prices")].set_value("Enter a figure").run()
    app.toggle[0].set_value(False).run()
    app.toggle[1].set_value(False).run()
    assert not app.exception
