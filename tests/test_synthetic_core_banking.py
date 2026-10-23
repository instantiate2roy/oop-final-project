import pytest

from src.csv_import import CsvImport
from src.synthetic_core_banking import GeneratorSettings, SyntheticCoreBanking

from conftest import ROOT


def generate(**settings):
    return SyntheticCoreBanking(GeneratorSettings(**settings), kb_folder=ROOT / "kb").generate()


@pytest.fixture(scope="module")
def small():
    return generate(customers=40)


def uuids(rows):
    return {row["uuid"] for row in rows}


def test_same_seed_gives_the_same_data(small):
    assert generate(customers=40) == small


def test_different_seed_gives_different_data(small):
    assert generate(customers=40, seed=7)["customer"] != small["customer"]


def test_every_reference_points_to_a_row_that_exists(small):
    customers, cases = uuids(small["customer"]), uuids(small["case"])
    contacts, promises = uuids(small["contact"]), uuids(small["ptp"])

    for name in ("address", "contact", "savings_history"):
        assert {row["customer_uuid"] for row in small[name]} <= customers, name
    assert {row["customer_uuid"] for row in small["case"]} <= customers
    for name in ("payment_history", "security", "contact_history", "ptp", "treatment_history", "case_snapshot"):
        assert {row["case_uuid"] for row in small[name]} <= cases, name
    assert {row["ptp_uuid"] for row in small["payment_history"]} - {""} <= promises
    assert {row["contact_uuid"] for row in small["contact_history"]} - {"", None} <= contacts


def test_case_balances_add_up(small):
    for case in small["case"]:
        assert int(case["outstanding_balance"]) == max(0, int(case["total_payable"]) - int(case["amount_paid"]))
        assert int(case["dpd"]) >= 0
        if case["account_status"] == "CLOSED":
            assert case["closed_date"]


def test_mia_is_arrears_over_the_monthly_instalment(small):
    """Whatever the repayment frequency, MIA counts months of repayments in arrears."""
    for case in small["case"]:
        monthly = int(case["total_payable"]) / int(case["term_months"])
        assert float(case["mia"]) == round(int(case["arrears_amount"]) / monthly, 2)


def test_nothing_happens_after_the_business_date(small):
    as_of = GeneratorSettings().as_of.isoformat()
    assert max(row["payment_date"] for row in small["payment_history"]) <= as_of
    assert max(row["contact_datetime"][:10] for row in small["contact_history"]) <= as_of
    assert max(row["snapshot_date"] for row in small["case_snapshot"]) <= as_of


def test_written_files_load_back(tmp_path):
    counts = SyntheticCoreBanking(GeneratorSettings(customers=20), kb_folder=ROOT / "kb").write(tmp_path)
    loaded = CsvImport().load_dir_data(tmp_path)

    assert {name: len(table["rows"]) for name, table in loaded.items()} == counts


def test_data_folder_matches_the_generator(tmp_path):
    """data/ is exactly what the notebook's REGENERATE cell would write, so the two can't drift apart."""
    SyntheticCoreBanking(GeneratorSettings(seed=2026), kb_folder=ROOT / "kb").write(tmp_path)

    for path in sorted((ROOT / "data").glob("*.csv")):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes(), path.name
