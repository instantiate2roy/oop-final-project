from datetime import datetime

import pandas as pd
import pytest

from src.compliance import ComplianceDecision, ProposedAction, RuleOutcome
from src.display.notification_list import NotificationList
from src.exceptions import NotificationError
from src.notifications import MessageTemplate, NotificationLog

RUN_AT = datetime(2026, 10, 23, 9, 0)
LOANS = pd.DataFrame([{"case_uuid": "k1", "customer_uuid": "c1", "account_no": "LN-1", "customer": "Okot Peter",
                       "dpd": 15.0, "arrears_amount": 295500.0, "outstanding_balance": 1200000.0}])
CUSTOMERS = [{"uuid": "c1", "given_name": "Peter"}]
CONTACTS = [
    {"customer_uuid": "c1", "contact_type_id": "MOBILE_PHONE", "value": "+256700000001", "valid": "N"},
    {"customer_uuid": "c1", "contact_type_id": "MOBILE_PHONE", "value": "+256700000002", "valid": "Y"},
    {"customer_uuid": "c1", "contact_type_id": "EMAIL", "value": "peter@example.com", "valid": "Y"},
]


def decision(treatment="SMS", allowed=True, case="k1"):
    outcomes = [RuleOutcome("Case status", True)]
    if not allowed:
        outcomes.append(RuleOutcome("Minimum gap", False, "last SMS was 2.0 hours ago (minimum 24)"))
    return ComplianceDecision(ProposedAction(case, "c1", treatment, RUN_AT), allowed, outcomes)


@pytest.fixture
def log():
    return NotificationLog(LOANS, CUSTOMERS, CONTACTS)


def test_message_template_is_abstract():
    with pytest.raises(TypeError):
        MessageTemplate()


@pytest.mark.parametrize("treatment, recipient, status, words", [
    ("SMS", "+256700000002", "SENT", "Dear Peter, your loan LN-1 is 15 days overdue. Please pay UGX 295,500"),
    ("EMAIL", "peter@example.com", "SENT", "UGX 1,200,000 outstanding"),
    ("CALL", "+256700000002", "QUEUED", "Call Okot Peter about loan LN-1"),
    ("DCA", "Debt collection agency", "REFERRED", "Refer loan LN-1 (Okot Peter) to Debt collection agency"),
    ("LEGAL", "Legal department", "REFERRED", "to Legal department"),
])
def test_each_channel_addresses_and_words_its_message(log, treatment, recipient, status, words):
    entry = log.record(decision(treatment))

    assert (entry.channel, entry.recipient, entry.status, entry.reason) == (treatment, recipient, status, "")
    assert words in entry.message
    assert entry.created_at == RUN_AT


def test_blocked_action_is_logged_with_its_reason(log):
    entry = log.record(decision(allowed=False))

    assert entry.status == "BLOCKED"
    assert entry.reason == "Minimum gap: last SMS was 2.0 hours ago (minimum 24)"
    assert "Dear Peter" in entry.message                    # what would have been sent


def test_unknown_channel_or_case(log):
    with pytest.raises(NotificationError, match="No message template for the PIGEON channel"):
        log.record(decision("PIGEON"))
    with pytest.raises(NotificationError, match="isn't in the portfolio"):
        log.record(decision(case="k9"))


def test_customer_without_contacts_gets_a_blank_recipient():
    entry = NotificationLog(LOANS, CUSTOMERS, []).record(decision("SMS"))
    assert entry.recipient == ""


def test_log_as_a_table(log):
    assert list(log.to_frame().columns)[:3] == ["created_at", "case_uuid", "account_no"]
    assert log.to_frame().empty

    log.record_all([decision("SMS"), decision("CALL", allowed=False)])

    assert list(log.to_frame()["status"]) == ["SENT", "BLOCKED"]


def test_notification_list_shows_the_log(log):
    log.record_all([decision("SMS"), decision("DCA")])
    shown = NotificationList(log).matches()

    assert len(shown) == 2
    assert shown.iloc[0]["created_at"] == "2026-10-23 09:00"


def test_daily_run_logs_every_decision(today, policy, data):
    from src.compliance import ComplianceChecker
    from src.dunning import DunningEngine

    actions = DunningEngine(policy, data["contact"]["rows"]).propose(today, RUN_AT)
    checker = ComplianceChecker(policy, data["contact_history"]["rows"], data["case"]["rows"], data["ptp"]["rows"])
    checker.check_all(actions)
    log = NotificationLog(today, data["customer"]["rows"], data["contact"]["rows"])

    log.record_all(checker.decisions)
    frame = log.to_frame()

    assert len(frame) == len(actions)
    assert (frame["recipient"] != "").all()                 # dunning only proposes reachable channels
    assert (frame["status"] == "BLOCKED").sum() == sum(not d.allowed for d in checker.decisions)
