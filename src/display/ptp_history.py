import pandas as pd

from .filters import ChoiceFilter, RangeFilter, TextFilter
from .history_list import HistoryList


class PtpHistory(HistoryList):
    """Every promise to pay, latest promised date first, with how much of it was paid."""

    title = "Promise-to-pay history"
    columns = ["created_date", "promised_date", "account_no", "customer", "promised_amount",
               "paid_amount", "paid_percent", "status"]
    sort_by = ["promised_date", "account_no"]
    sort_ascending = [False, True]
    noun = "promises"

    def __init__(self, ptp_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.ptp_rows = ptp_rows
        super().__init__(case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        promises = pd.DataFrame(self.ptp_rows)
        promises["promised_amount"] = pd.to_numeric(promises["promised_amount"]).astype(int)
        promises["paid_amount"] = pd.to_numeric(promises["paid_amount"]).astype(int)
        # share of the promise that was paid; more than 100 when the customer paid extra
        promises["paid_percent"] = (100 * promises["paid_amount"] / promises["promised_amount"]).round().astype(int)
        promises["month"] = pd.to_datetime(promises["promised_date"]).dt.strftime("%Y-%m")
        return self.add_account_and_customer(promises)

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("month", "Promised in", width="400px"),
            ChoiceFilter("status", "Status", width="400px"),
            RangeFilter("promised_amount", "Promised", step=10_000, width="400px"),
            RangeFilter("paid_percent", "Paid %", width="400px"),
        ]
