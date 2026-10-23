import json

import pytest

from src.exceptions import PolicyError
from src.policy import Policy


@pytest.mark.parametrize("dpd, bucket", [
    (0, "current"), (1, "1-30"), (30, "1-30"), (31, "31-60"), (60, "31-60"),
    (61, "61-90"), (90, "61-90"), (91, "90+"), (5000, "90+"),
])
def test_bucket_edges(policy, dpd, bucket):
    assert policy.bucket_for(dpd) == bucket


def test_negative_dpd_is_rejected(policy):
    with pytest.raises(PolicyError, match="negative"):
        policy.bucket_for(-1)


@pytest.mark.parametrize("facts, profile", [
    ({"bucket": "90+", "reachability": 0.2}, "high_risk_unreachable"),
    ({"bucket": "90+", "reachability": 0.3}, "high_risk_unreachable"),
    ({"bucket": "90+", "reachability": 0.31}, "high_risk"),
    ({"bucket": "61-90", "reachability": None}, "high_risk"),       # no contact attempts: not "unreachable"
    ({"bucket": "1-30", "broken_ptps": 2}, "broken_promises"),
    ({"bucket": "1-30", "broken_ptps": 1}, "low_risk"),
    ({"bucket": "31-60", "outstanding_ratio": 0.5}, "medium_risk_large_balance"),
    ({"bucket": "31-60", "outstanding_ratio": 0.49}, "medium_risk"),
    ({"bucket": "current"}, "current"),
])
def test_profiles_are_checked_in_priority_order(policy, facts, profile):
    assert policy.profile_for(facts) == profile


def test_profiles_are_sorted_by_priority(policy_data):
    policy_data["profiles"].reverse()
    priorities = [rule["priority"] for rule in Policy(policy_data).profiles]
    assert priorities == sorted(priorities)


def test_treatment_for(policy):
    assert policy.treatment_for("high_risk", "90+") == "DCA"
    assert policy.treatment_for("low_risk", "90+") is None
    assert policy.treatment_for("unknown", "1-30") is None


def test_load_missing_file_and_bad_json(tmp_path):
    with pytest.raises(PolicyError, match="not found"):
        Policy.load(tmp_path / "policy.json")
    bad = tmp_path / "policy.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(PolicyError, match="not valid JSON"):
        Policy.load(bad)


def test_load_checks_treatment_codes_against_the_kb(tmp_path, policy_data):
    policy_data["treatment_matrix"]["low_risk"]["1-30"] = "PIGEON"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy_data), encoding="utf-8")

    assert Policy.load(path).treatment_for("low_risk", "1-30") == "PIGEON"     # no kb: not checked
    with pytest.raises(PolicyError, match="unknown treatment 'PIGEON'"):
        Policy.load(path, treatment_type_rows=[{"type_id": "SMS"}, {"type_id": "CALL"}])


def broken(policy_data, change):
    change(policy_data)
    return policy_data


@pytest.mark.parametrize("change, message", [
    (lambda p: p.pop("compliance"), "missing the 'compliance' section"),
    (lambda p: p["buckets"]["edges"][0].update(lower=1), "must start at 0"),
    (lambda p: p["buckets"]["edges"][2].update(lower=32), "leave a gap or overlap"),
    (lambda p: p["buckets"]["edges"][2].update(lower=30), "leave a gap or overlap"),
    (lambda p: p["buckets"]["edges"][1].update(upper=None), "Only the last bucket can be open-ended"),
    (lambda p: p["buckets"]["edges"][-1].update(upper=999), "must have no upper limit"),
    (lambda p: p["profiles"][1].update(name="high_risk_unreachable"), "same name"),
    (lambda p: p["profiles"][1].update(priority=1), "same priority"),
    (lambda p: p["profiles"][0]["conditions"].update(shoe_size={"max": 9}), "unknown field 'shoe_size'"),
    (lambda p: p["profiles"][0]["conditions"].update(dpd={"above": 9}), "unknown test"),
    (lambda p: p["profiles"][0]["conditions"].update(dpd={"min": 9, "max": 1}), "min is above max"),
    (lambda p: p["profiles"][0]["conditions"].update(bucket={"in": ["120+"]}), "unknown buckets"),
    (lambda p: p["treatment_matrix"].update(nobody={"1-30": "SMS"}), "unknown profile 'nobody'"),
    (lambda p: p["treatment_matrix"]["low_risk"].update({"120+": "SMS"}), r"unknown bucket '120\+'"),
    (lambda p: p["compliance"]["contact_hours"].update(start="18:00"), "must be before they end"),
    (lambda p: p["compliance"]["contact_hours"].update(days=["Mon", "Funday"]), "Unknown contact days"),
    (lambda p: p["compliance"]["channels"]["SMS"].update(max_per_day=9), "max per day"),
])
def test_mistakes_in_the_policy_are_rejected(policy_data, change, message):
    with pytest.raises(PolicyError, match=message):
        Policy(broken(policy_data, change))


@pytest.mark.parametrize("pd_value, tier", [
    (0.0, "A"), (0.0099, "A"), (0.01, "B"), (0.029, "B"), (0.03, "C"), (0.1, "D"), (0.499, "D"), (0.5, "E"), (1.0, "E"),
])
def test_tier_edges(policy, pd_value, tier):
    assert policy.tier_for(pd_value) == tier


@pytest.mark.parametrize("pd_value", [-0.01, 1.01])
def test_pd_outside_zero_to_one_is_rejected(policy, pd_value):
    with pytest.raises(PolicyError, match="between 0 and 1"):
        policy.tier_for(pd_value)


def test_profile_rules_can_test_the_tier(policy_data):
    policy_data["profiles"][0]["conditions"] = {"tier": {"in": ["E"]}}
    policy = Policy(policy_data)

    assert policy.profile_for({"tier": "E", "bucket": "1-30"}) == "high_risk_unreachable"


@pytest.mark.parametrize("change, message", [
    (lambda p: p.pop("risk_tiers"), "missing the 'risk_tiers' section"),
    (lambda p: p.update(risk_tiers=[]), "At least one risk tier"),
    (lambda p: p["risk_tiers"][1].update(name="A"), "same name"),
    (lambda p: p["risk_tiers"][1].update(max_pd=None), "Only the last risk tier can be open-ended"),
    (lambda p: p["risk_tiers"][-1].update(max_pd=1.0), "must have no max_pd"),
    (lambda p: p["risk_tiers"][1].update(max_pd=0.005), "must rise from tier to tier"),
    (lambda p: p["risk_tiers"][3].update(max_pd=1.5), "between 0 and 1"),
])
def test_mistakes_in_the_risk_tiers_are_rejected(policy_data, change, message):
    with pytest.raises(PolicyError, match=message):
        Policy(broken(policy_data, change))


def test_compliance_tables(policy):
    channels, general = policy.compliance_tables()

    assert list(channels["Channel"]) == ["SMS", "EMAIL", "CALL"]
    assert "SMS → EMAIL" in general.set_index("Rule").loc["Fallback treatments", "Setting"]


@pytest.mark.parametrize("change, message", [
    (lambda p: p["buckets"]["edges"][1].pop("upper"), "Bucket 2 is missing 'upper'"),
    (lambda p: p["buckets"].pop("edges"), "buckets is missing 'edges'"),
    (lambda p: p["buckets"].update(edges={"name": "current"}), "buckets.edges must be a list"),
    (lambda p: p["risk_tiers"][0].pop("max_pd"), "Risk tier 1 is missing 'max_pd'"),
    (lambda p: p["profiles"][0].pop("conditions"), "Profile rule 1 is missing 'conditions'"),
    (lambda p: p["profiles"][0].update(conditions=["bucket"]), "conditions of profile 'high_risk_unreachable' must be an object"),
    (lambda p: p["profiles"][0]["conditions"].update(dpd=30), "condition on 'dpd' must be an object"),
    (lambda p: p["treatment_matrix"].update(low_risk="SMS"), "Treatment matrix row 'low_risk' must be an object"),
    (lambda p: p["compliance"].pop("contact_hours"), "compliance is missing 'contact_hours'"),
    (lambda p: p["compliance"]["contact_hours"].pop("days"), "compliance.contact_hours is missing 'days'"),
    (lambda p: p["compliance"]["channels"]["SMS"].pop("max_per_week"), "Channel limits for SMS is missing 'max_per_week'"),
    (lambda p: p["compliance"]["contact_hours"].update(days=5), "badly formed: TypeError"),     # the safety net
])
def test_missing_or_misshapen_settings_raise_policy_error(policy_data, change, message):
    with pytest.raises(PolicyError, match=message):
        Policy(broken(policy_data, change))


def test_policy_that_is_not_an_object():
    with pytest.raises(PolicyError, match="must be a JSON object, not list"):
        Policy([])


def test_raw_settings_are_private(policy):
    assert not hasattr(policy, "data")
    assert policy.default_profile == "current"
    assert policy.fallback_treatments == {"SMS": "EMAIL", "CALL": "SMS"}

    policy.fallback_treatments["SMS"] = "PIGEON"           # a copy: changing it doesn't change the policy
    assert policy.fallback_treatments["SMS"] == "EMAIL"


def test_describe_conditions(policy):
    assert Policy.describe({"bucket": {"in": ["61-90", "90+"]}, "reachability": {"max": 0.3}}) == \
        "bucket in 61-90, 90+ and reachability ≤ 0.3"
