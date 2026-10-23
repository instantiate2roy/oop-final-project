"""
Shared fixtures. The real data files are loaded once per test run and must not be changed by tests;
tests that need odd data build their own small rows instead.
"""
import copy
import json
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")           # draw charts without a screen

from src.csv_import import CsvImport
from src.policy import Policy
from src.risk import FeatureBuilder, LogisticPdModel

ROOT = Path(__file__).resolve().parent.parent
POLICY_FILE = ROOT / "configs" / "policy.json"
AS_OF = date(2026, 10, 22)      # the business date the data files were generated for


@pytest.fixture(scope="session")
def kb():
    return CsvImport().load_dir_data(ROOT / "kb")


@pytest.fixture(scope="session")
def data():
    return CsvImport().load_dir_data(ROOT / "data")


@pytest.fixture(scope="session")
def policy(kb):
    return Policy.load(POLICY_FILE, treatment_type_rows=kb["treatment_type"]["rows"])


@pytest.fixture
def policy_data():
    """A fresh copy of the policy file's contents, safe to change in a test."""
    return copy.deepcopy(json.loads(POLICY_FILE.read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def builder(data, kb):
    return FeatureBuilder(data, kb, as_of=AS_OF)


@pytest.fixture(scope="session")
def training_rows(builder):
    return builder.training_set()


@pytest.fixture(scope="session")
def trained_model(training_rows):
    return LogisticPdModel(FeatureBuilder.NUMERIC, FeatureBuilder.CATEGORICAL).fit(training_rows)


@pytest.fixture(scope="session")
def portfolio(policy, trained_model, builder, data):
    from src.profiling import Profiler
    return Profiler(policy, trained_model, builder, data).portfolio()


@pytest.fixture(scope="session")
def roll_rates(policy, data):
    from src.risk import RollRateModel
    return RollRateModel(policy).fit(data["case_snapshot"]["rows"])


@pytest.fixture(scope="session")
def today(portfolio):
    return portfolio[portfolio["business_date"] == pd.Timestamp(AS_OF)]
