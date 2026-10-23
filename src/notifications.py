"""
The messages the collections run sends, and the in-system log that records them.

Each channel has a MessageTemplate that knows who the message goes to and what it says. The
NotificationLog turns every compliance decision into a Notification: sent (or queued for an officer,
or referred) when compliance allowed it, and kept with the blocking reason when it didn't, so the log
shows what the system did and what it held back.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from datetime import datetime

import pandas as pd

from .compliance import ComplianceDecision
from .exceptions import NotificationError


@dataclass(frozen=True)
class Notification:
    created_at: datetime
    case_uuid: str
    account_no: str
    customer: str
    channel: str            # SMS, EMAIL, CALL, DCA or LEGAL
    recipient: str          # phone number, email address, or who the case is referred to
    message: str
    status: str             # SENT, QUEUED (a call for an officer to make), REFERRED or BLOCKED
    reason: str = ""        # why compliance blocked it


class MessageTemplate(ABC):
    """How one channel addresses and words a message about an overdue loan."""

    channel: str = ""
    status: str = "SENT"            # what an allowed message becomes

    @abstractmethod
    def recipient(self, loan: dict) -> str:
        """Who the message goes to."""

    @abstractmethod
    def render(self, loan: dict) -> str:
        """The message text, from the loan's details (account_no, given_name, dpd, arrears_amount, ...)."""


class SmsTemplate(MessageTemplate):
    channel = "SMS"

    def recipient(self, loan):
        return loan["mobile"]

    def render(self, loan):
        return (f"Dear {loan['given_name']}, your loan {loan['account_no']} is {loan['dpd']} days overdue. "
                f"Please pay UGX {loan['arrears_amount']:,} to bring it up to date, or call us if you need help.")


class EmailTemplate(MessageTemplate):
    channel = "EMAIL"

    def recipient(self, loan):
        return loan["email"]

    def render(self, loan):
        return (f"Subject: Loan {loan['account_no']} is overdue\n\n"
                f"Dear {loan['given_name']},\n\nYour loan {loan['account_no']} is {loan['dpd']} days overdue, "
                f"with UGX {loan['arrears_amount']:,} in arrears and UGX {loan['outstanding_balance']:,} "
                f"outstanding. Please pay the arrears, or contact your loan officer to agree a plan.")


class CallTemplate(MessageTemplate):
    """A call is queued for an officer to make, with a note of what to discuss."""

    channel = "CALL"
    status = "QUEUED"

    def recipient(self, loan):
        return loan["mobile"]

    def render(self, loan):
        return (f"Call {loan['customer']} about loan {loan['account_no']}: {loan['dpd']} days overdue, "
                f"UGX {loan['arrears_amount']:,} in arrears. Ask for payment or a promise to pay.")


class ReferralTemplate(MessageTemplate):
    """A referral hands the case to someone outside the collections team; the customer isn't contacted."""

    status = "REFERRED"

    def __init__(self, channel: str, referred_to: str):
        self.channel = channel
        self.referred_to = referred_to

    def recipient(self, loan):
        return self.referred_to

    def render(self, loan):
        return (f"Refer loan {loan['account_no']} ({loan['customer']}) to {self.referred_to}: "
                f"{loan['dpd']} days overdue, UGX {loan['outstanding_balance']:,} outstanding.")


class NotificationLog:
    """
    The in-system log of every message the collections run sent, queued, referred or held back.

    loans: the portfolio rows for the run's business date (Profiler.portfolio() for one date).
    The first valid mobile number and email address on record are used, as in the dunning engine.
    """

    TEMPLATES = [SmsTemplate(), EmailTemplate(), CallTemplate(),
                 ReferralTemplate("DCA", "Debt collection agency"), ReferralTemplate("LEGAL", "Legal department")]

    def __init__(self, loans: pd.DataFrame, customer_rows: list[dict], contact_rows: list[dict],
                 templates: list[MessageTemplate] | None = None):
        self.templates = {template.channel: template for template in (templates or self.TEMPLATES)}

        contacts = pd.DataFrame(contact_rows, columns=["customer_uuid", "contact_type_id", "value", "valid"])
        valid = contacts[contacts["valid"] == "Y"]
        first = lambda kind: valid[valid["contact_type_id"] == kind].groupby("customer_uuid")["value"].first()
        given_names = {row["uuid"]: row["given_name"] for row in customer_rows}

        details = loans[["case_uuid", "customer_uuid", "account_no", "customer", "dpd", "arrears_amount",
                         "outstanding_balance"]].copy()
        for column in ["dpd", "arrears_amount", "outstanding_balance"]:
            details[column] = details[column].astype(int)
        details["given_name"] = details["customer_uuid"].map(given_names).fillna(details["customer"])
        details["mobile"] = details["customer_uuid"].map(first("MOBILE_PHONE")).fillna("")
        details["email"] = details["customer_uuid"].map(first("EMAIL")).fillna("")
        self._loans = details.set_index("case_uuid").to_dict("index")
        self.entries: list[Notification] = []

    def record(self, decision: ComplianceDecision) -> Notification:
        """Write one compliance decision to the log as a notification."""
        action = decision.action
        template = self.templates.get(action.treatment)
        if template is None:
            raise NotificationError(f"No message template for the {action.treatment} channel")
        loan = self._loans.get(action.case_uuid)
        if loan is None:
            raise NotificationError(f"Case {action.case_uuid} isn't in the portfolio for this run")
        entry = Notification(
            created_at=action.planned_at, case_uuid=action.case_uuid, account_no=loan["account_no"],
            customer=loan["customer"], channel=template.channel, recipient=template.recipient(loan),
            message=template.render(loan), status=template.status if decision.allowed else "BLOCKED",
            reason="" if decision.allowed else f"{decision.blocked_by}: {decision.reason}")
        self.entries.append(entry)
        return entry

    def record_all(self, decisions: list[ComplianceDecision]) -> list[Notification]:
        return [self.record(decision) for decision in decisions]

    def to_frame(self) -> pd.DataFrame:
        """The log as a table, one row per notification."""
        return pd.DataFrame([asdict(entry) for entry in self.entries],
                            columns=[f.name for f in fields(Notification)])
