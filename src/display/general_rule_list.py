import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import TextFilter


class GeneralRuleList(Display):
    """The compliance rules that apply to every contact, whatever the channel."""

    title = "General compliance rules"
    columns = ["rule", "setting"]
    sort_by = ["order"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        _, general = self.policy.compliance_tables()
        # fallbacks are shown in the treatment matrix instead
        general = general[general["Rule"] != "Fallback treatments"]
        return pd.DataFrame([{"order": position, "rule": row["Rule"], "setting": str(row["Setting"])}
                             for position, row in enumerate(general.to_dict("records"))])

    def build_filters(self):
        return [
            TextFilter("rule", "Rule", width="400px"),
            TextFilter("setting", "Setting", width="400px"),
        ]