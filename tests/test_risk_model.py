import joblib
import numpy as np
import pandas as pd
import pytest

from src.exceptions import ModelError
from src.risk import FeatureBuilder, LogisticPdModel, RiskModel

from conftest import AS_OF


def toy_frame(rows: int = 400, seed: int = 1) -> pd.DataFrame:
    """A frame where defaults rise with x and don't depend on the colour."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    return pd.DataFrame({"x": x, "colour": rng.choice(["red", "blue"], rows),
                         "defaulted": (x + rng.normal(scale=0.5, size=rows) > 1).astype(int)})


def toy_model() -> LogisticPdModel:
    return LogisticPdModel(["x"], ["colour"])


# ---- FeatureBuilder ----------------------------------------------------------

def test_feature_builder_needs_the_history_files(data, kb):
    with pytest.raises(ModelError, match="case_snapshot"):
        FeatureBuilder({name: data[name] for name in data if name != "case_snapshot"}, kb, as_of=AS_OF)


def test_training_set_only_uses_months_whose_outcome_is_known(training_rows):
    known_until = pd.Timestamp(AS_OF) - pd.DateOffset(months=6)

    assert training_rows["observed_on"].max() <= known_until
    assert (training_rows["dpd"] < 90).all()                    # loans already in default aren't scored
    assert set(training_rows["defaulted"]) == {0, 1}


def test_training_set_features_are_complete(training_rows):
    assert not training_rows[FeatureBuilder.NUMERIC + FeatureBuilder.CATEGORICAL].isna().any().any()


def test_label_is_the_worst_dpd_in_the_next_six_snapshots(training_rows, builder):
    snapshots = builder.snapshots.sort_values("snapshot_date")
    for row in training_rows.sample(50, random_state=0).itertuples():
        later = snapshots[(snapshots["case_uuid"] == row.case_uuid) & (snapshots["snapshot_date"] > row.observed_on)]
        assert row.defaulted == int(later["dpd"].head(6).max() >= 90 if len(later) else 0)


def test_broken_promises_only_count_promises_due_by_that_date(training_rows, builder):
    broken = builder.ptps[builder.ptps["status"] == "BROKEN"].merge(
        builder.cases[["uuid", "customer_uuid"]], left_on="case_uuid", right_on="uuid")
    for row in training_rows.sample(50, random_state=1).itertuples():
        mine = broken[(broken["customer_uuid"] == row.customer_uuid) & (broken["promised_date"] <= row.observed_on)]
        assert row.broken_ptps_to_date == len(mine)


def test_current_features_cover_every_open_loan(builder, data):
    today = builder.current_features()
    open_cases = {row["uuid"] for row in data["case"]["rows"] if row["account_status"] == "OPEN"}

    assert set(today["case_uuid"]) == open_cases
    assert (today["observed_on"] == pd.Timestamp(AS_OF)).all()


# ---- LogisticPdModel and RiskModel -------------------------------------------

def test_risk_model_is_abstract():
    with pytest.raises(TypeError):
        RiskModel(["x"])


def test_untrained_model_cannot_predict_or_be_saved(tmp_path):
    with pytest.raises(ModelError, match="hasn't been trained"):
        toy_model().predict_pd(toy_frame())
    with pytest.raises(ModelError, match="hasn't been trained"):
        toy_model().save(tmp_path / "model.joblib")


def test_training_needs_both_outcomes():
    frame = toy_frame().assign(defaulted=0)
    with pytest.raises(ModelError, match="both defaults and non-defaults"):
        toy_model().fit(frame)


def test_missing_feature_columns_are_reported():
    with pytest.raises(ModelError, match="Missing feature columns: colour"):
        toy_model().fit(toy_frame().drop(columns="colour"))


def test_predictions_are_probabilities_on_the_same_rows():
    frame = toy_frame()
    model = toy_model().fit(frame)

    predictions = model.predict_pd(frame.iloc[10:20])

    assert list(predictions.index) == list(range(10, 20))
    assert predictions.between(0, 1).all()


def test_coefficients_show_which_way_a_feature_moves_risk():
    table = toy_model().fit(toy_frame()).coefficients()

    assert table.iloc[0]["feature"] == "x"                   # the strongest feature comes first
    assert table.iloc[0]["effect"] == "raises risk"


def test_unseen_category_does_not_break_scoring():
    model = toy_model().fit(toy_frame())
    assert model.predict_pd(toy_frame(5).assign(colour="green")).between(0, 1).all()


def test_evaluation_bands(trained_model, training_rows):
    evaluation = trained_model.evaluate(training_rows, bands=5)

    assert list(evaluation.bands["band"]) == ["A", "B", "C", "D", "E"]
    assert evaluation.bands["loans"].sum() == evaluation.rows == len(training_rows)
    assert evaluation.defaults == training_rows["defaulted"].sum()
    rates = list(evaluation.bands["default_rate"])
    assert rates[-1] == max(rates)                          # band E is the riskiest


def test_evaluation_needs_both_outcomes(trained_model, training_rows):
    with pytest.raises(ModelError, match="both defaults and non-defaults"):
        trained_model.evaluate(training_rows[training_rows["defaulted"] == 0])


def test_model_ranks_later_months_it_never_saw(training_rows):
    """The notebook's time split. A floor rather than an exact value, so small library changes don't fail it."""
    split = pd.Timestamp("2025-12-31")
    train, test = training_rows[training_rows["observed_on"] < split], training_rows[training_rows["observed_on"] >= split]

    model = LogisticPdModel(FeatureBuilder.NUMERIC, FeatureBuilder.CATEGORICAL).fit(train)

    assert model.evaluate(test).auc > 0.8


def test_save_and_load_give_the_same_predictions(tmp_path):
    frame = toy_frame()
    model = toy_model().fit(frame)

    path = model.save(tmp_path / "models" / "pd_model.joblib")
    loaded = RiskModel.load(path)

    assert isinstance(loaded, LogisticPdModel)
    pd.testing.assert_series_equal(loaded.predict_pd(frame), model.predict_pd(frame))


def test_load_rejects_missing_and_foreign_files(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        RiskModel.load(tmp_path / "missing.joblib")
    joblib.dump({"not": "a model"}, tmp_path / "other.joblib")
    with pytest.raises(ModelError, match="doesn't contain a risk model"):
        RiskModel.load(tmp_path / "other.joblib")
