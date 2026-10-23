import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import TextFilter


class RiskTierList(Display):
    """The risk tiers from the policy, from the lowest probability of default to the highest."""

    title = "Risk tiers"
    columns = ["tier", "description", "from_pd", "to_pd"]
    sort_by = ["order"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        rows, lower = [], 0.0
        for position, tier in enumerate(self.policy.risk_tiers):
            upper = tier["max_pd"]
            rows.append({"order": position, "tier": tier["name"], "description": tier.get("description", ""),
                         "from_pd": f"{lower:.0%}", "to_pd": "100%" if upper is None else f"under {upper:.0%}"})
            lower = upper
        return pd.DataFrame(rows)

    def build_filters(self):
        return [
            TextFilter("tier", "Tier", width="400px"),
            TextFilter("description", "Description", width="400px"),
        ]
