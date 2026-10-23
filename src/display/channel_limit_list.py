import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import ChoiceFilter


class ChannelLimitList(Display):
    """How often each channel may be used to contact the same customer."""

    title = "Channel limits"
    columns = ["channel", "max_per_day", "max_per_week", "min_gap_hours"]
    sort_by = ["order"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "order": position,
            "channel": channel,
            "max_per_day": limits["max_per_day"],
            "max_per_week": limits["max_per_week"],
            "min_gap_hours": limits["min_gap_hours"],
        } for position, (channel, limits) in enumerate(self.policy.compliance["channels"].items())])

    def build_filters(self):
        return [
            ChoiceFilter("channel", "Channel", width="400px"),
        ]