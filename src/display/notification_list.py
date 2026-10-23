import pandas as pd

from ..notifications import NotificationLog
from .display import Display
from .filters import ChoiceFilter, TextFilter


class NotificationList(Display):
    """The notification log: every message the collections run sent, queued, referred or held back."""

    title = "Notification log"
    columns = ["created_at", "account_no", "customer", "channel", "recipient", "status", "reason", "message"]
    sort_by = ["created_at", "account_no"]
    sort_ascending = [False, True]

    def __init__(self, log: NotificationLog, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.log = log
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        frame = self.log.to_frame()
        frame["created_at"] = pd.to_datetime(frame["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        return frame

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("channel", "Channel", width="400px"),
            ChoiceFilter("status", "Status", width="400px"),
            TextFilter("reason", "Reason", width="400px"),
            TextFilter("message", "Message", width="400px"),
        ]
