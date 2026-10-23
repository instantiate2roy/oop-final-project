import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import RangeFilter, TextFilter


class BucketList(Display):
    """The delinquency buckets from the policy, in order from current to most overdue."""

    title = "Delinquency buckets"
    columns = ["bucket", "from_dpd", "to_dpd"]
    sort_by = ["from_dpd"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "bucket": bucket["name"],
            "from_dpd": bucket["lower"],
            "to_dpd": "no limit" if bucket["upper"] is None else str(bucket["upper"]),
        } for bucket in self.policy.buckets])

    def build_filters(self):
        return [
            TextFilter("bucket", "Bucket", width="400px"),
            RangeFilter("from_dpd", "From DPD", width="400px"),
        ]