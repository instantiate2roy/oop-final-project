from datetime import datetime

import pandas as pd

from .compliance.contact_rules import ProposedAction
from .policy import Policy


class DunningEngine:
    """
    Decides which treatment each overdue loan should get, from its profile and bucket.

    When a customer has no valid contact for the chosen channel (no mobile for an SMS or call,
    no email for an email), the policy's fallback treatment is used instead. A referral (DCA or
    LEGAL) that is already running isn't proposed again.
    """

    REFERRALS = {"DCA", "LEGAL"}

    def __init__(self, policy: Policy, contact_rows: list[dict]):
        self.policy = policy
        contacts = pd.DataFrame(contact_rows)
        valid = contacts[contacts["valid"] == "Y"]
        self.has_mobile = set(valid.loc[valid["contact_type_id"] == "MOBILE_PHONE", "customer_uuid"])
        self.has_email = set(valid.loc[valid["contact_type_id"] == "EMAIL", "customer_uuid"])

    def _reachable_by(self, treatment: str, customer: str) -> bool:
        if treatment in ("SMS", "CALL"):
            return customer in self.has_mobile
        if treatment == "EMAIL":
            return customer in self.has_email
        return True                                     # referrals don't need a contact

    def propose(self, today: pd.DataFrame, planned_at: datetime) -> list[ProposedAction]:
        """One action for each overdue loan whose profile and bucket have a treatment."""
        fallbacks = self.policy.fallback_treatments
        actions = []
        for row in today[today["dpd"] > 0].to_dict("records"):
            treatment = self.policy.treatment_for(row["profile"], row["bucket"])
            tried = set()
            while treatment and not self._reachable_by(treatment, row["customer_uuid"]) and treatment not in tried:
                tried.add(treatment)
                treatment = fallbacks.get(treatment)
            already_referred = treatment in self.REFERRALS and treatment == row.get("current_treatment")
            if treatment and self._reachable_by(treatment, row["customer_uuid"]) and not already_referred:
                actions.append(ProposedAction(row["case_uuid"], row["customer_uuid"], treatment, planned_at,
                                              row["profile"], row["bucket"]))
        return actions
