import pandas as pd

from .case_history import CaseHistory
from .filters import ChoiceFilter, RangeFilter, TextFilter


class PtpHistory(CaseHistory):
    """Every promise to pay, newest promised date first, with how much of it was paid."""

    title = "Promise-to-pay history"
    columns = ["created_date", "promised_date", "account_no", "customer", "promised_amount",
               "paid_amount", "paid_percent", "status"]
    sort_by = ["promised_date", "account_no"]
    sort_ascending = [False, True]

    def prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        history["promised_amount"] = pd.to_numeric(history["promised_amount"]).astype(int)
        history["paid_amount"] = pd.to_numeric(history["paid_amount"]).astype(int)
        # share of the promise that was paid; more than 100 when the customer paid extra
        history["paid_percent"] = (100 * history["paid_amount"] / history["promised_amount"]).round().astype(int)
        return self.add_month(history, "promised_date")

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("month", "Promised in", width="400px"),
            ChoiceFilter("status", "Status", width="400px"),
            RangeFilter("promised_amount", "Promised", step=10_000, width="400px"),
            RangeFilter("paid_percent", "Paid %", width="400px"),
        ]