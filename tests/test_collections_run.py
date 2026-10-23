"""The daily collections run: profiling, the dunning engine and the compliance checks."""
from datetime import date, datetime

import pandas as pd
import pytest

from src.compliance import (CaseStatusRule, ComplianceChecker, ContactHoursRule, DailyChannelLimitRule,
                            DailyContactCapRule, MinimumGapRule, PromisePendingRule, ProposedAction,
                            WeeklyChannelLimitRule)
from src.dunning import DunningEngine
from src.profiling import Profiler

from conftest import AS_OF

MONDAY_10AM = datetime(2026, 10, 19, 10, 0)
OPEN_CASE = {"status": "IN_PROGRESS", "ptp_pending_until": None}


def action(treatment="SMS", planned_at=MONDAY_10AM, case="k1", customer="c1"):
    return ProposedAction(case, customer, treatment, planned_at)


def history(*contacts):
    """(datetime text, channel) pairs as the rules expect them."""
    return pd.DataFrame({"contact_datetime": pd.to_datetime([when for when, _ in contacts]),
                         "channel": [channel for _, channel in contacts]})


NO_HISTORY = history()


# ---- compliance rules ----------------------------------------------------------

def test_case_status_rule(policy):
    rule = CaseStatusRule(policy)
    assert rule.check(action(), NO_HISTORY, OPEN_CASE).allowed
    blocked = rule.check(action(), NO_HISTORY, {"status": "WRITTEN_OFF"})
    assert (blocked.allowed, blocked.reason) == (False, "case is WRITTEN_OFF")


def test_promise_pending_rule_holds_until_the_promised_date(policy):
    rule = PromisePendingRule(policy)
    promised = OPEN_CASE | {"ptp_pending_until": date(2026, 10, 19)}

    assert not rule.check(action(), NO_HISTORY, promised).allowed
    assert rule.check(action(planned_at=datetime(2026, 10, 20, 10)), NO_HISTORY, promised).allowed


@pytest.mark.parametrize("planned_at, treatment, allowed", [
    (MONDAY_10AM, "SMS", True),
    (datetime(2026, 10, 19, 8, 0), "CALL", True),
    (datetime(2026, 10, 19, 7, 59), "CALL", False),
    (datetime(2026, 10, 19, 17, 59), "EMAIL", True),
    (datetime(2026, 10, 19, 18, 0), "EMAIL", False),
    (datetime(2026, 10, 25, 10, 0), "SMS", False),          # Sunday
    (datetime(2026, 10, 25, 22, 0), "DCA", True),           # referrals don't contact the customer
])
def test_contact_hours_rule(policy, planned_at, treatment, allowed):
    assert ContactHoursRule(policy).check(action(treatment, planned_at), NO_HISTORY, OPEN_CASE).allowed is allowed


def test_daily_channel_limit_counts_the_same_channel_only(policy):
    rule = DailyChannelLimitRule(policy)            # SMS: 1 a day
    assert not rule.check(action("SMS"), history(("2026-10-19 08:30", "SMS")), OPEN_CASE).allowed
    assert rule.check(action("SMS"), history(("2026-10-19 08:30", "CALL")), OPEN_CASE).allowed
    assert rule.check(action("SMS"), history(("2026-10-18 08:30", "SMS")), OPEN_CASE).allowed


def test_weekly_channel_limit(policy):
    rule = WeeklyChannelLimitRule(policy)           # SMS: 3 a week
    three = history(("2026-10-13 11:00", "SMS"), ("2026-10-15 09:00", "SMS"), ("2026-10-17 09:00", "SMS"))
    assert not rule.check(action("SMS"), three, OPEN_CASE).allowed
    assert rule.check(action("SMS", datetime(2026, 10, 20, 12)), three, OPEN_CASE).allowed      # oldest one aged out


def test_minimum_gap(policy):
    rule = MinimumGapRule(policy)                   # CALL: 4 hours apart
    assert not rule.check(action("CALL"), history(("2026-10-19 07:00", "CALL")), OPEN_CASE).allowed
    assert rule.check(action("CALL"), history(("2026-10-19 06:00", "CALL")), OPEN_CASE).allowed
    assert rule.check(action("CALL"), NO_HISTORY, OPEN_CASE).allowed


def test_daily_contact_cap_counts_every_channel(policy):
    rule = DailyContactCapRule(policy)              # 2 contacts a day
    two = history(("2026-10-19 08:10", "SMS"), ("2026-10-19 08:20", "EMAIL"))
    assert not rule.check(action("CALL"), two, OPEN_CASE).allowed
    assert rule.check(action("LEGAL"), two, OPEN_CASE).allowed


# ---- compliance checker --------------------------------------------------------

CASE_ROWS = [{"uuid": "k1", "customer_uuid": "c1", "status": "IN_PROGRESS"},
             {"uuid": "k2", "customer_uuid": "c1", "status": "IN_PROGRESS"},
             {"uuid": "k3", "customer_uuid": "c2", "status": "SETTLED"}]
PTP_ROWS = [{"uuid": "p1", "case_uuid": "k2", "promised_date": "2026-10-25", "status": "PENDING"},
            {"uuid": "p2", "case_uuid": "k1", "promised_date": "2026-10-30", "status": "BROKEN"}]


def checker(policy, contacts=(("2026-01-05 09:00", "SMS"),)):
    """By default one old contact, long enough ago not to affect any rule."""
    rows = [{"case_uuid": "k1", "contact_datetime": when, "channel": channel} for when, channel in contacts]
    return ComplianceChecker(policy, rows, CASE_ROWS, PTP_ROWS)


def test_checker_works_before_anyone_has_been_contacted(policy):
    assert checker(policy, contacts=()).check(action()).allowed


def test_checker_works_without_any_promises(policy):
    decision = ComplianceChecker(policy, [], CASE_ROWS, []).check(action(case="k2"))
    assert decision.allowed


def test_checker_stops_at_the_first_rule_that_blocks(policy):
    decision = checker(policy).check(action(case="k3", customer="c2"))

    assert not decision.allowed
    assert (decision.blocked_by, len(decision.outcomes)) == ("Case status", 1)


def test_checker_only_holds_for_pending_promises(policy):
    run = checker(policy)
    assert run.check(action(case="k2")).blocked_by == "Promise pending"
    assert run.check(action(case="k1")).allowed                 # k1's promise was broken, not pending


def test_checker_counts_contacts_allowed_earlier_in_the_same_run(policy):
    run = checker(policy, [("2026-10-19 08:00", "CALL")])

    first = run.check(action("EMAIL", MONDAY_10AM))
    second = run.check(action("SMS", MONDAY_10AM))

    assert first.allowed
    assert second.blocked_by == "Daily contact cap"


def test_check_all_returns_one_row_per_action(policy):
    table = checker(policy).check_all([action(), action(case="k3", customer="c2")])

    assert list(table["allowed"]) == [True, False]
    assert list(table["blocked_by"]) == ["", "Case status"]


# ---- dunning engine ------------------------------------------------------------

CONTACTS = [
    {"customer_uuid": "mobile", "contact_type_id": "MOBILE_PHONE", "valid": "Y"},
    {"customer_uuid": "email", "contact_type_id": "EMAIL", "valid": "Y"},
    {"customer_uuid": "bad", "contact_type_id": "MOBILE_PHONE", "valid": "N"},
]


def propose(policy, *loans):
    today = pd.DataFrame([{"case_uuid": f"k{i}", "current_treatment": ""} | loan for i, loan in enumerate(loans)])
    return [a.treatment for a in DunningEngine(policy, CONTACTS).propose(today, MONDAY_10AM)]


def test_dunning_uses_the_treatment_matrix(policy):
    assert propose(policy, {"customer_uuid": "mobile", "dpd": 5, "profile": "low_risk", "bucket": "1-30"}) == ["SMS"]


def test_dunning_falls_back_when_the_channel_cannot_reach_the_customer(policy):
    # CALL -> SMS -> EMAIL: no valid mobile, but a valid email
    loan = {"customer_uuid": "email", "dpd": 40, "profile": "medium_risk_large_balance", "bucket": "31-60"}
    assert propose(policy, loan) == ["EMAIL"]


def test_dunning_skips_customers_it_cannot_reach(policy):
    assert propose(policy, {"customer_uuid": "bad", "dpd": 5, "profile": "low_risk", "bucket": "1-30"}) == []


def test_dunning_skips_loans_that_are_not_overdue_or_have_no_treatment(policy):
    assert propose(policy, {"customer_uuid": "mobile", "dpd": 0, "profile": "low_risk", "bucket": "current"},
                   {"customer_uuid": "mobile", "dpd": 5, "profile": "current", "bucket": "1-30"}) == []


def test_dunning_does_not_refer_a_loan_twice(policy):
    referral = {"customer_uuid": "bad", "dpd": 100, "profile": "high_risk", "bucket": "90+"}
    assert propose(policy, referral) == ["DCA"]                                 # referrals need no contact
    assert propose(policy, referral | {"current_treatment": "DCA"}) == []


# ---- profiler ------------------------------------------------------------------

@pytest.mark.parametrize("mia, bucket", [(0, "current"), (0.5, "1-30"), (1.0, "1-30"), (1.1, "31-60"), (3.5, "90+")])
def test_mia_bucket(policy, trained_model, builder, data, mia, bucket):
    assert Profiler(policy, trained_model, builder, data).mia_bucket(mia) == bucket


def test_portfolio_has_one_row_per_open_loan_today(portfolio, data):
    today = portfolio[portfolio["business_date"] == pd.Timestamp(AS_OF)]
    open_cases = [row for row in data["case"]["rows"] if row["account_status"] == "OPEN"]

    assert len(today) == len(open_cases)
    assert today["case_uuid"].is_unique


def test_portfolio_profiles_follow_the_policy(portfolio, policy):
    for row in portfolio.sample(200, random_state=0).to_dict("records"):
        assert row["bucket"] == policy.bucket_for(int(row["dpd"]))
        assert row["tier"] == policy.tier_for(row["pd"])
        facts = {"bucket": row["bucket"], "dpd": row["dpd"], "mia": row["mia"],
                 "outstanding_balance": row["outstanding_balance"], "outstanding_ratio": row["outstanding_ratio"],
                 "arrears_amount": row["arrears_amount"], "reachability": row["reachability"],
                 "broken_ptps": row["broken_ptps"], "pd": row["pd"], "tier": row["tier"]}
        assert row["profile"] == policy.profile_for(facts)


def test_reachability_is_a_share_or_unknown(portfolio):
    known = portfolio["reachability"].dropna()
    assert known.between(0, 1).all()
    assert portfolio["reachability"].isna().any()               # some customers had no contact attempts


def test_daily_run_end_to_end(portfolio, policy, data):
    today = portfolio[portfolio["business_date"] == pd.Timestamp(AS_OF)]
    actions = DunningEngine(policy, data["contact"]["rows"]).propose(today, datetime(2026, 10, 23, 9, 0))
    decisions = ComplianceChecker(policy, data["contact_history"]["rows"], data["case"]["rows"],
                                  data["ptp"]["rows"]).check_all(actions)

    assert len(actions) > 0
    assert len(decisions) == len(actions)
    assert decisions["allowed"].any()
