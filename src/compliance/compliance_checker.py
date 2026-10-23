from dataclasses import dataclass, field

import pandas as pd

from .contact_rules import (CaseStatusRule, ContactHoursRule, ContactRule, DailyChannelLimitRule,
                            DailyContactCapRule, MinimumGapRule, ProposedAction, PromisePendingRule,
                            RuleOutcome, WeeklyChannelLimitRule)


@dataclass(frozen=True)
class ComplianceDecision:
    action: ProposedAction
    allowed: bool
    outcomes: list[RuleOutcome] = field(default_factory=list)

    @property
    def blocked_by(self) -> str:
        """The first rule that blocked the action, or '' if it was allowed."""
        return next((o.rule for o in self.outcomes if not o.allowed), "")

    @property
    def reason(self) -> str:
        return next((o.reason for o in self.outcomes if not o.allowed), "")


class ComplianceChecker:
    """
    Runs every compliance rule on each proposed treatment, in order, and records the result.

    An allowed contact is added to the customer's history straight away, so a second action
    for the same customer in the same run sees it (e.g. the daily cap counts it).
    """

    def __init__(self, policy, contact_history_rows: list[dict], case_rows: list[dict],
                 ptp_rows: list[dict], rules: list[ContactRule] | None = None):
        self.rules = rules or [rule(policy) for rule in (
            CaseStatusRule, PromisePendingRule, ContactHoursRule, DailyContactCapRule,
            DailyChannelLimitRule, WeeklyChannelLimitRule, MinimumGapRule)]

        cases = pd.DataFrame(case_rows)
        # naming the columns keeps an empty history or promise list usable, e.g. before anyone is contacted
        history = pd.DataFrame(contact_history_rows, columns=["case_uuid", "contact_datetime", "channel"]).merge(
            cases[["uuid", "customer_uuid"]], left_on="case_uuid", right_on="uuid", suffixes=("", "_case"))
        history["contact_datetime"] = pd.to_datetime(history["contact_datetime"])
        self._history = {customer: rows[["contact_datetime", "channel"]].sort_values("contact_datetime")
                         for customer, rows in history.groupby("customer_uuid")}

        pending = pd.DataFrame(ptp_rows, columns=["case_uuid", "promised_date", "status"])
        pending = pending[pending["status"] == "PENDING"]
        latest_promise = pd.to_datetime(pending.groupby("case_uuid")["promised_date"].max()).dt.date.to_dict()
        self._cases = {row["uuid"]: {"status": row["status"], "ptp_pending_until": latest_promise.get(row["uuid"])}
                       for row in case_rows}
        self.decisions: list[ComplianceDecision] = []

    def check(self, action: ProposedAction) -> ComplianceDecision:
        history = self._history.get(action.customer_uuid,
                                    pd.DataFrame({"contact_datetime": pd.to_datetime([]), "channel": []}))
        outcomes = []
        for rule in self.rules:
            outcome = rule.check(action, history, self._cases[action.case_uuid])
            outcomes.append(outcome)
            if not outcome.allowed:
                break                     # the first rule that blocks decides; later rules aren't needed
        decision = ComplianceDecision(action, all(o.allowed for o in outcomes), outcomes)
        if decision.allowed and action.treatment in ("SMS", "EMAIL", "CALL"):
            new = pd.DataFrame({"contact_datetime": [pd.Timestamp(action.planned_at)], "channel": [action.treatment]})
            self._history[action.customer_uuid] = pd.concat([history, new], ignore_index=True)
        self.decisions.append(decision)
        return decision

    def check_all(self, actions: list[ProposedAction]) -> pd.DataFrame:
        """Check every action and return one row per decision."""
        return pd.DataFrame([self._row(self.check(action)) for action in actions])

    @staticmethod
    def _row(decision: ComplianceDecision) -> dict:
        a = decision.action
        return {"case_uuid": a.case_uuid, "customer_uuid": a.customer_uuid, "profile": a.profile,
                "bucket": a.bucket, "treatment": a.treatment, "planned_at": a.planned_at,
                "allowed": decision.allowed, "blocked_by": decision.blocked_by, "reason": decision.reason}
