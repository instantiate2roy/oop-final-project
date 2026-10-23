import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import TextFilter


class ProfileRuleList(Display):
    """
    The profile rules from the policy, in the order they are checked.

    The first rule whose conditions all match decides a customer's profile;
    anyone no rule matches gets the default profile, listed last.
    """

    title = "Profile rules"
    columns = ["profile", "description", "conditions", "priority"]
    sort_by = ["order"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        rows = [{
            "order": position,
            "priority": str(rule["priority"]),
            "profile": rule["name"],
            "description": rule.get("description", ""),
            "conditions": self.policy.describe(rule["conditions"]),
        } for position, rule in enumerate(self.policy.profiles)]
        rows.append({"order": len(rows), "priority": "default", "profile": self.policy.default_profile,
                     "description": "Anyone no rule matches", "conditions": "-"})
        return pd.DataFrame(rows)

    def build_filters(self):
        return [
            TextFilter("profile", "Profile", width="400px"),
            TextFilter("conditions", "Conditions", width="400px"),
            TextFilter("description", "Description", width="400px"),
        ]