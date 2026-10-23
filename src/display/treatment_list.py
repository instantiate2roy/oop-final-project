import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import ChoiceFilter


class TreatmentList(Display):
    """
    The treatment matrix from the policy, one row per profile and bucket that has a treatment.

    'fallback' is the treatment used instead when the main one can't be,
    for example when a customer has no valid mobile number for an SMS.
    """

    title = "Treatment matrix"
    columns = ["profile", "bucket", "treatment", "treatment_name", "fallback"]
    sort_by = ["profile_order", "bucket_order"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3,
                 treatment_type_rows: list[dict] | None = None):
        self.policy = policy
        # optional lookup of treatment names, e.g. SMS -> "SMS reminders"
        self.treatment_names = {row["type_id"]: row["name"] for row in (treatment_type_rows or [])}
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        fallbacks = self.policy.data.get("fallback_treatments", {})
        rows = []
        for profile_order, rule in enumerate(self.policy.profiles):
            for bucket_order, bucket in enumerate(self.policy.buckets):
                treatment = self.policy.treatment_for(rule["name"], bucket["name"])
                if treatment:
                    rows.append({
                        "profile_order": profile_order, "bucket_order": bucket_order,
                        "profile": rule["name"], "bucket": bucket["name"], "treatment": treatment,
                        "treatment_name": self.treatment_names.get(treatment, ""),
                        "fallback": fallbacks.get(treatment, "-"),
                    })
        return pd.DataFrame(rows)

    def build_filters(self):
        return [
            ChoiceFilter("profile", "Profile", width="400px"),
            ChoiceFilter("bucket", "Bucket", width="400px"),
            ChoiceFilter("treatment", "Treatment", width="400px"),
        ]