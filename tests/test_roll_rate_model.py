import pandas as pd
import pytest

from src.exceptions import ModelError
from src.risk import RollRateModel


def snapshot(case, month, dpd, balance, status="OPEN", final=""):
    return {"case_uuid": case, "snapshot_date": month, "dpd": str(dpd), "outstanding_balance": str(balance),
            "account_status": status, "status": final or ("CURRENT" if status == "OPEN" else "")}


# three loans over three month-ends, with balances chosen so the roll rates are easy to check
SNAPSHOTS = [
    snapshot("a", "2026-01-31", 0, 600), snapshot("a", "2026-02-28", 0, 500), snapshot("a", "2026-03-31", 10, 400),
    snapshot("b", "2026-01-31", 0, 200), snapshot("b", "2026-02-28", 10, 200),
    snapshot("b", "2026-03-31", 40, 200, "CLOSED", "WRITTEN_OFF"),
    snapshot("c", "2026-01-31", 10, 100), snapshot("c", "2026-02-28", 0, 0, "CLOSED", "SETTLED"),
]


@pytest.fixture
def model(policy):
    return RollRateModel(policy).fit(SNAPSHOTS)


def test_states_are_the_buckets_then_the_closed_states(policy):
    assert RollRateModel(policy).states == ["current", "1-30", "31-60", "61-90", "90+", "SETTLED", "WRITTEN_OFF"]


def test_roll_rates_are_weighted_by_balance(model):
    # current: a 600 -> current, b 200 -> 1-30, a 500 -> 1-30
    assert model.matrix.loc["current", "current"] == pytest.approx(600 / 1300)
    assert model.matrix.loc["current", "1-30"] == pytest.approx(700 / 1300)
    # 1-30: c 100 -> SETTLED, b 200 -> WRITTEN_OFF
    assert model.matrix.loc["1-30", "SETTLED"] == pytest.approx(100 / 300)
    assert model.matrix.loc["1-30", "WRITTEN_OFF"] == pytest.approx(200 / 300)
    assert model.transitions == 5


def test_every_row_adds_up_and_closed_states_stay_closed(model):
    assert model.matrix.sum(axis=1).to_numpy() == pytest.approx(1)
    assert model.matrix.loc["SETTLED", "SETTLED"] == model.matrix.loc["WRITTEN_OFF", "WRITTEN_OFF"] == 1
    assert model.matrix.loc["61-90", "61-90"] == 1            # no history for this bucket: stays put


def test_fit_until_ignores_later_months(policy):
    early = RollRateModel(policy).fit(SNAPSHOTS, until="2026-02-28")
    assert early.transitions == 3
    assert early.matrix.loc["1-30", "SETTLED"] == 1


def test_projection_keeps_the_total_and_moves_it_monthly(model):
    start = pd.Series({"current": 1000.0})

    projection = model.project(start, months=3)

    assert list(projection.index) == [0, 1, 2, 3]
    assert projection.sum(axis=1).to_numpy() == pytest.approx(1000)
    assert projection.loc[1, "1-30"] == pytest.approx(1000 * 700 / 1300)
    assert projection.loc[3, "WRITTEN_OFF"] > projection.loc[2, "WRITTEN_OFF"] > 0


def test_expected_write_offs_count_only_new_ones(model):
    table = model.expected_write_offs(pd.Series({"current": 1000.0, "WRITTEN_OFF": 50.0}), horizons=(1, 2))

    assert list(table["months ahead"]) == [1, 2]
    assert table.loc[0, "written off (UGX)"] == 0                 # nothing reaches WRITTEN_OFF in one month
    assert table.loc[1, "written off (UGX)"] == round(1000 * 700 / 1300 * 200 / 300)
    assert table.loc[1, "share of open balance %"] == round(100 * 700 / 1300 * 200 / 300, 1)


def test_balances_by_state(model):
    loans = pd.DataFrame({"bucket": ["current", "current", "90+"], "outstanding_balance": [10, 20, 5]})
    assert model.balances(loans).to_dict() == {"current": 30, "1-30": 0, "31-60": 0, "61-90": 0, "90+": 5,
                                               "SETTLED": 0, "WRITTEN_OFF": 0}


def test_actual_path_follows_each_loan_with_its_starting_balance(model):
    path = model.actual_path(SNAPSHOTS, "2026-01-31", months=2)

    assert path.loc[0].to_dict() == pytest.approx({"current": 800, "1-30": 100, "31-60": 0, "61-90": 0, "90+": 0,
                                                   "SETTLED": 0, "WRITTEN_OFF": 0})
    assert path.loc[2, "SETTLED"] == 100                     # c closed in month 1 and stays settled
    assert (path.loc[2, "1-30"], path.loc[2, "WRITTEN_OFF"]) == (600, 200)


@pytest.mark.parametrize("call, message", [
    (lambda m: m.project(pd.Series({"current": 1.0}), months=0), "at least 1"),
    (lambda m: m.project(pd.Series({"lost": 1.0})), "Unknown states"),
    (lambda m: m.actual_path(SNAPSHOTS, "2026-01-15", 1), "No snapshots on"),
    (lambda m: m.actual_path(SNAPSHOTS, "2026-02-28", 2), "Only 1 months of history"),
])
def test_bad_requests(model, call, message):
    with pytest.raises(ModelError, match=message):
        call(model)


def test_unfitted_model_and_odd_data(policy):
    with pytest.raises(ModelError, match="hasn't been fitted"):
        RollRateModel(policy).project(pd.Series({"current": 1.0}))
    with pytest.raises(ModelError, match="unknown final status"):
        RollRateModel(policy).fit(SNAPSHOTS + [snapshot("d", "2026-01-31", 0, 1, "CLOSED", "SOLD")])
    with pytest.raises(ModelError, match="No month-to-month moves"):
        RollRateModel(policy).fit(SNAPSHOTS[:1])


def test_real_history(roll_rates, data):
    assert roll_rates.transitions > 10_000
    assert roll_rates.matrix.loc["current", "current"] > 0.8             # most current balance stays current
    assert roll_rates.matrix.loc["90+", "WRITTEN_OFF"] > roll_rates.matrix.loc["current", "WRITTEN_OFF"]


def test_back_test_is_close_at_six_months(policy, data):
    """The notebook's back-test: fit up to September 2025, project, and compare with what happened."""
    rows = data["case_snapshot"]["rows"]
    start_date = "2025-09-30"
    model = RollRateModel(policy).fit(rows, until=start_date)
    snaps = model.states_of(rows)
    open_then = snaps[(snaps["snapshot_date"] == start_date) & ~snaps["state"].isin(model.CLOSED_STATES)]
    start = model.balances(open_then, state_column="state")

    projected, actual = model.project(start, 6), model.actual_path(rows, start_date, 6)

    assert actual.sum(axis=1).to_numpy() == pytest.approx(start.sum())
    for state in ["90+", "WRITTEN_OFF"]:
        assert abs(projected.loc[6, state] - actual.loc[6, state]) / start.sum() < 0.02
