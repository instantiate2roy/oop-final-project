"""
The compliance rules a proposed treatment must pass before it is carried out.

Each rule is a ContactRule with one method, check(), which says whether the action is allowed
and, if not, why. The limits themselves come from the policy file, so the rules only hold logic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

# treatments that contact the customer directly; referrals (DCA, LEGAL) don't
CONTACT_CHANNELS = {"SMS", "EMAIL", "CALL"}


@dataclass(frozen=True)
class ProposedAction:
    """A treatment the dunning step wants to carry out."""
    case_uuid: str
    customer_uuid: str
    treatment: str              # SMS, EMAIL, CALL, DCA or LEGAL
    planned_at: datetime
    profile: str = ""
    bucket: str = ""


@dataclass(frozen=True)
class RuleOutcome:
    rule: str
    allowed: bool
    reason: str = ""


class ContactRule(ABC):
    """One compliance rule. Subclasses read their limits from the policy in __init__."""

    name: str = "rule"

    @abstractmethod
    def check(self, action: ProposedAction, history: pd.DataFrame, case: dict) -> RuleOutcome:
        """
        history: the customer's past contacts (columns contact_datetime, channel), newest last.
        case: the case's current details (status, pending promise, ...).
        """

    def _allow(self) -> RuleOutcome:
        return RuleOutcome(self.name, True)

    def _block(self, reason: str) -> RuleOutcome:
        return RuleOutcome(self.name, False, reason)


class CaseStatusRule(ContactRule):
    name = "Case status"

    def __init__(self, policy):
        self.blocked = set(policy.compliance["do_not_treat_case_statuses"])

    def check(self, action, history, case):
        if case["status"] in self.blocked:
            return self._block(f"case is {case['status']}")
        return self._allow()


class PromisePendingRule(ContactRule):
    name = "Promise pending"

    def __init__(self, policy):
        self.active = policy.compliance["hold_while_ptp_pending"]

    def check(self, action, history, case):
        if self.active and case.get("ptp_pending_until") and action.planned_at.date() <= case["ptp_pending_until"]:
            return self._block(f"promise to pay due {case['ptp_pending_until']}")
        return self._allow()


class ContactHoursRule(ContactRule):
    name = "Contact hours"

    def __init__(self, policy):
        hours = policy.compliance["contact_hours"]
        self.start, self.end, self.days = hours["start"], hours["end"], set(hours["days"])

    def check(self, action, history, case):
        if action.treatment not in CONTACT_CHANNELS:
            return self._allow()
        when = action.planned_at
        if when.strftime("%a") not in self.days:
            return self._block(f"{when:%A} is not a contact day")
        if not self.start <= when.strftime("%H:%M") < self.end:
            return self._block(f"{when:%H:%M} is outside {self.start}-{self.end}")
        return self._allow()


class _ChannelRule(ContactRule):
    """Shared set-up for the rules that limit one channel."""

    def __init__(self, policy):
        self.limits = policy.compliance["channels"]

    def _same_channel(self, action, history):
        return history[history["channel"] == action.treatment]


class DailyChannelLimitRule(_ChannelRule):
    name = "Daily channel limit"

    def check(self, action, history, case):
        if action.treatment not in self.limits:
            return self._allow()
        today = self._same_channel(action, history)
        today = today[today["contact_datetime"].dt.date == action.planned_at.date()]
        limit = self.limits[action.treatment]["max_per_day"]
        if len(today) >= limit:
            return self._block(f"{action.treatment} already used {len(today)} time(s) today (limit {limit})")
        return self._allow()


class WeeklyChannelLimitRule(_ChannelRule):
    name = "Weekly channel limit"

    def check(self, action, history, case):
        if action.treatment not in self.limits:
            return self._allow()
        week = self._same_channel(action, history)
        week = week[week["contact_datetime"] > action.planned_at - timedelta(days=7)]
        limit = self.limits[action.treatment]["max_per_week"]
        if len(week) >= limit:
            return self._block(f"{action.treatment} used {len(week)} times in 7 days (limit {limit})")
        return self._allow()


class MinimumGapRule(_ChannelRule):
    name = "Minimum gap"

    def check(self, action, history, case):
        if action.treatment not in self.limits:
            return self._allow()
        previous = self._same_channel(action, history)
        if previous.empty:
            return self._allow()
        gap = (action.planned_at - previous["contact_datetime"].max()).total_seconds() / 3600
        needed = self.limits[action.treatment]["min_gap_hours"]
        if gap < needed:
            return self._block(f"last {action.treatment} was {gap:.1f} hours ago (minimum {needed})")
        return self._allow()


class DailyContactCapRule(ContactRule):
    name = "Daily contact cap"

    def __init__(self, policy):
        self.cap = policy.compliance["max_contacts_per_day"]

    def check(self, action, history, case):
        if action.treatment not in CONTACT_CHANNELS:
            return self._allow()
        today = history[history["contact_datetime"].dt.date == action.planned_at.date()]
        if len(today) >= self.cap:
            return self._block(f"customer already contacted {len(today)} time(s) today (cap {self.cap})")
        return self._allow()
