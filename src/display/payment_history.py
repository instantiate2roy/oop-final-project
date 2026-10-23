import pandas as pd

from .case_history import CaseHistory
from .filters import ChoiceFilter, RangeFilter, TextFilter


class PaymentHistory(CaseHistory):
    """Every payment received, newest first, showing whether it was made against a promise to pay."""

    title = "Payment history"
    columns = ["payment_date", "account_no", "customer", "amount", "channel", "reference",
               "status", "against_ptp"]
    sort_by = ["payment_date", "account_no"]
    sort_ascending = [False, True]

    def prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        history["amount"] = pd.to_numeric(history["amount"]).astype(int)
        history["against_ptp"] = history["ptp_uuid"].ne("").map({True: "Y", False: "N"})
        return self.add_month(history, "payment_date")

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


    