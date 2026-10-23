"""The lists shown in the notebook: built from the real data files, and from small rows for the edge cases."""
from datetime import date

import pandas as pd
import pytest

from src.display.bucket_list import BucketList
from src.display.case_list import CaseList
from src.display.channel_limit_list import ChannelLimitList
from src.display.contact_history import ContactHistory
from src.display.customer_list import CustomerList
from src.display.general_rule_list import GeneralRuleList
from src.display.payment_history import PaymentHistory
from src.display.profile_rule_list import ProfileRuleList
from src.display.risk_tier_list import RiskTierList
from src.display.ptp_history import PtpHistory
from src.display.treatment_history import TreatmentHistory
from src.display.treatment_list import TreatmentList
from src.display.worklist import Worklist
from src.exceptions import DisplayError

from conftest import AS_OF

CUSTOMERS = [{"uuid": "c1", "surname": "Okot", "given_name": "Peter", "sex": "M", "dob": "2000-10-22", "nin": "N1"},
             {"uuid": "c2", "surname": "Apio", "given_name": "Grace", "sex": "F", "dob": "2000-10-23", "nin": "N2"}]
CASES = [{"uuid": "k1", "account_no": "LN-1", "customer_uuid": "c1"},
         {"uuid": "k2", "account_no": "LN-2", "customer_uuid": "gone"}]


# ---- the real data -----------------------------------------------------------

def test_case_list_keeps_every_case(data, kb):
    cases = CaseList(data["case"]["rows"], data["customer"]["rows"], kb["loan_types"]["rows"])

    assert len(cases.frame) == len(data["case"]["rows"])
    dpd = list(cases.matches()["dpd"])
    assert dpd == sorted(dpd, reverse=True)                 # most overdue first


def test_customer_list_keeps_every_customer(data, kb):
    customers = CustomerList(data["customer"]["rows"], AS_OF, address_rows=data["address"]["rows"],
                             contact_rows=data["contact"]["rows"], location_rows=kb["locations"]["rows"])

    assert len(customers.frame) == len(data["customer"]["rows"])
    assert customers.frame["age"].between(18, 70).all()
    assert (customers.frame["district"] != "").mean() > 0.9


@pytest.mark.parametrize("make, file", [
    (lambda d: ContactHistory(d["contact_history"]["rows"], d["case"]["rows"], d["customer"]["rows"],
                              contact_rows=d["contact"]["rows"]), "contact_history"),
    (lambda d: PaymentHistory(d["payment_history"]["rows"], d["case"]["rows"], d["customer"]["rows"]), "payment_history"),
    (lambda d: PtpHistory(d["ptp"]["rows"], d["case"]["rows"], d["customer"]["rows"]), "ptp"),
    (lambda d: TreatmentHistory(d["treatment_history"]["rows"], d["case"]["rows"], d["customer"]["rows"],
                                as_of=AS_OF), "treatment_history"),
])
def test_history_lists_keep_every_row_and_find_the_customer(data, make, file):
    history = make(data)

    assert len(history.frame) == len(data[file]["rows"])
    assert (history.frame["customer"] != history.MISSING_CUSTOMER).all()
    first = history.matches().iloc[0][history.sort_by[0]]
    assert first == history.frame[history.sort_by[0]].max()          # newest first


def test_policy_lists_match_the_policy(policy, kb):
    assert len(BucketList(policy).frame) == len(policy.buckets)
    assert len(ProfileRuleList(policy).frame) == len(policy.profiles) + 1           # plus the default profile
    assert list(ChannelLimitList(policy).matches()["channel"]) == list(policy.compliance["channels"])
    assert "Fallback treatments" not in set(GeneralRuleList(policy).frame["rule"])

    treatments = TreatmentList(policy, treatment_type_rows=kb["treatment_type"]["rows"])
    assert len(treatments.frame) == sum(len(row) for row in policy.treatment_matrix.values())
    sms = treatments.frame[treatments.frame["treatment"] == "SMS"].iloc[0]
    assert (sms["treatment_name"], sms["fallback"]) == ("SMS reminders", "EMAIL")


# ---- edge cases --------------------------------------------------------------

def test_age_counts_the_birthday_on_the_business_date():
    ages = CustomerList(CUSTOMERS, date(2026, 10, 22)).frame.set_index("uuid")["age"]

    assert ages["c1"] == 26           # birthday today
    assert ages["c2"] == 25           # birthday tomorrow


def test_customer_without_contacts_or_addresses_gets_blanks():
    frame = CustomerList(CUSTOMERS, AS_OF).frame

    assert list(frame["contacts"]) == [0, 0]
    assert list(frame["phone"]) == ["", ""]


def test_customer_list_picks_the_best_contacts_and_address():
    contacts = [
        {"customer_uuid": "c1", "contact_type_id": "MOBILE_PHONE", "value": "+256700000001", "valid": "N"},
        {"customer_uuid": "c1", "contact_type_id": "MOBILE_PHONE", "value": "+256700000002", "valid": "Y"},
        {"customer_uuid": "c1", "contact_type_id": "EMAIL", "value": "peter@example.com", "valid": "Y"},
    ]
    addresses = [
        {"customer_uuid": "c1", "address_type_id": "HOME", "address_line": "Old Road", "status": "MOVED", "location_uuid": "l1"},
        {"customer_uuid": "c1", "address_type_id": "HOME", "address_line": "New Road", "status": "VERIFIED", "location_uuid": "l1"},
        {"customer_uuid": "c1", "address_type_id": "WORK", "address_line": "Office", "status": "VERIFIED", "location_uuid": "l1"},
    ]
    locations = [{"uuid": "l1", "village": "Kireka", "district": "Wakiso"}]

    peter = CustomerList(CUSTOMERS, AS_OF, contact_rows=contacts, address_rows=addresses,
                         location_rows=locations).frame.set_index("uuid").loc["c1"]

    assert (peter["phone"], peter["email"], peter["contacts"]) == ("+256700000002", "peter@example.com", 2)
    assert (peter["home_address"], peter["district"]) == ("New Road, Kireka", "Wakiso")


def test_case_list_flags_a_missing_customer_or_loan_type():
    cases = [case | {"loan_type_uuid": loan_type, "dpd": "0", "mia": "0", "outstanding_balance": "100",
                     "arrears_amount": "0", "status": "CURRENT", "next_due_date": ""}
             for case, loan_type in zip(CASES, ["gone", "t1"])]
    loan_types = [{"uuid": "t1", "name": "Business loan"}]

    frame = CaseList(cases, CUSTOMERS, loan_types).frame.set_index("account_no")

    assert (frame.loc["LN-1", "customer"], frame.loc["LN-1", "loan_type"]) == ("Okot Peter", CaseList.MISSING_LOAN_TYPE)
    assert (frame.loc["LN-2", "customer"], frame.loc["LN-2", "loan_type"]) == (CaseList.MISSING_CUSTOMER, "Business loan")


def test_history_row_for_an_unknown_case_is_an_error():
    contacts = [{"uuid": "h1", "case_uuid": "nope", "contact_uuid": "", "contact_datetime": "2026-01-01 09:00",
                 "channel": "SMS", "outcome": "SMS_DELIVERED", "notes": ""}]

    with pytest.raises(DisplayError, match="1 contacts refer to cases that don't exist"):
        ContactHistory(contacts, CASES, CUSTOMERS)


def test_history_row_for_an_unknown_customer_is_flagged():
    payments = [{"uuid": "p1", "case_uuid": "k2", "ptp_uuid": "", "payment_date": "2026-01-01", "amount": "5000",
                 "channel": "CASH", "reference": "R1", "status": "POSTED"}]

    frame = PaymentHistory(payments, CASES, CUSTOMERS).frame

    assert frame.loc[0, "customer"] == PaymentHistory.MISSING_CUSTOMER
    assert frame.loc[0, "against_ptp"] == "N"


def test_paid_percent_can_go_over_100():
    promises = [{"uuid": "t1", "case_uuid": "k1", "created_date": "2026-01-01", "promised_date": "2026-01-08",
                 "promised_amount": "20000", "paid_amount": "30000", "status": "KEPT"}]

    assert PtpHistory(promises, CASES, CUSTOMERS).frame.loc[0, "paid_percent"] == 150


def treatment(**changes):
    row = {"uuid": "t1", "case_uuid": "k1", "treatment_type_id": "SMS", "start_date": "2026-10-01",
           "end_date": "", "dpd_at_start": "3", "arrears_at_start": "5000", "assigned_to": "", "status": "ACTIVE"}
    return row | changes


def test_running_treatment_is_counted_to_the_business_date():
    frame = TreatmentHistory([treatment()], CASES, CUSTOMERS, as_of=date(2026, 10, 22),
                             treatment_type_rows=[{"type_id": "SMS", "name": "SMS reminders"}]).frame

    assert frame.loc[0, "days"] == 21
    assert (frame.loc[0, "end_date"], frame.loc[0, "assigned_to"]) == ("running", "unassigned")
    assert frame.loc[0, "treatment"] == "SMS reminders"


def test_treatment_that_ends_before_it_starts_is_an_error():
    with pytest.raises(DisplayError, match="end before they start"):
        TreatmentHistory([treatment(end_date="2026-09-01")], CASES, CUSTOMERS, as_of=AS_OF)


# ---- worklist and risk tiers ---------------------------------------------------

def test_worklist_ranks_overdue_loans_by_expected_loss(portfolio, today):
    worklist = Worklist(portfolio)

    overdue = today[today["dpd"] > 0]
    assert worklist.business_date == pd.Timestamp(AS_OF)
    assert len(worklist.frame) == len(overdue)
    shown = worklist.matches()
    assert list(shown["rank"]) == list(range(1, len(shown) + 1))
    assert shown["expected_loss"].is_monotonic_decreasing
    top = shown.iloc[0]
    assert top["expected_loss"] == round(overdue.set_index("account_no").loc[top["account_no"], "pd"]
                                         * top["outstanding_balance"])


def test_worklist_rank_survives_filtering(portfolio):
    worklist = Worklist(portfolio)
    officer = worklist.filters[2]                       # Officer
    officer.widget.value = "In-house"

    shown = worklist.matches()

    assert set(shown["assigned_to"]) == {"In-house"}
    whole_list = worklist.frame.set_index("account_no")["rank"]
    assert list(shown["rank"]) == list(whole_list.loc[shown["account_no"]])
    assert shown["rank"].is_monotonic_increasing


def test_worklist_for_an_earlier_business_date(portfolio):
    earlier = sorted(portfolio["business_date"].unique())[-3]
    worklist = Worklist(portfolio, earlier)
    assert (worklist.frame["business_date"] == earlier).all()


def test_risk_tier_list(policy):
    tiers = RiskTierList(policy).matches()

    assert list(tiers["tier"]) == ["A", "B", "C", "D", "E"]
    assert (tiers.iloc[0]["from_pd"], tiers.iloc[0]["to_pd"]) == ("0%", "under 1%")
    assert (tiers.iloc[-1]["from_pd"], tiers.iloc[-1]["to_pd"]) == ("50%", "100%")
