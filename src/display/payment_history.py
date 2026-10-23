import pandas as pd

from .filters import ChoiceFilter, RangeFilter, TextFilter
from .history_list import HistoryList


class PaymentHistory(HistoryList):
    """Every payment received, newest first, showing whether it was made against a promise to pay."""

    title = "Payment history"
    columns = ["payment_date", "account_no", "customer", "amount", "channel", "reference",
               "status", "against_ptp"]
    sort_by = ["payment_date", "account_no"]
    sort_ascending = [False, True]
    noun = "payments"

    def __init__(self, payment_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.payment_rows = payment_rows
        super().__init__(case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        payments = pd.DataFrame(self.payment_rows)
        payments["amount"] = pd.to_numeric(payments["amount"]).astype(int)
        payments["against_ptp"] = payments["ptp_uuid"].ne("").map({True: "Y", False: "N"})
        payments["month"] = pd.to_datetime(payments["payment_date"]).dt.strftime("%Y-%m")
        return self.add_account_and_customer(payments)

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            TextFilter("reference", "Reference", width="400px"),
            ChoiceFilter("month", "Month", width="400px"),
            ChoiceFilter("channel", "Channel", width="400px"),
            ChoiceFilter("status", "Status", width="400px"),
            ChoiceFilter("against_ptp", "Against PTP", width="400px"),
            RangeFilter("amount", "Amount", step=10_000, width="400px"),
        ]
