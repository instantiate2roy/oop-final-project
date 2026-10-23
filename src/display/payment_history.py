import pandas as pd

from ..exceptions import DisplayError
from .display import Display
from .filters import ChoiceFilter, RangeFilter, TextFilter


class PaymentHistory(Display):
    """Every payment received, newest first, showing whether it was made against a promise to pay."""

    title = "Payment history"
    columns = ["payment_date", "account_no", "customer", "amount", "channel", "reference",
               "status", "against_ptp"]
    sort_by = ["payment_date", "account_no"]
    sort_ascending = [False, True]
    MISSING_CUSTOMER = "(customer not found)"

    def __init__(self, payment_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.payment_rows = payment_rows
        self.case_rows = case_rows
        self.customer_rows = customer_rows
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        payments = pd.DataFrame(self.payment_rows)
        payments["amount"] = pd.to_numeric(payments["amount"]).astype(int)
        payments["against_ptp"] = payments["ptp_uuid"].ne("").map({True: "Y", False: "N"})
        payments["month"] = pd.to_datetime(payments["payment_date"]).dt.strftime("%Y-%m")

        # account number and customer name, through the case
        cases = pd.DataFrame(self.case_rows)[["uuid", "account_no", "customer_uuid"]]
        payments = payments.merge(cases, how="left", left_on="case_uuid", right_on="uuid", suffixes=("", "_case"))
        if payments["account_no"].isna().any():
            raise DisplayError(f"{payments['account_no'].isna().sum()} payments refer to cases that don't exist")

        customers = pd.DataFrame(self.customer_rows)
        names = (customers["surname"] + " " + customers["given_name"]).rename("customer")
        payments = payments.merge(pd.concat([customers["uuid"], names], axis=1), how="left",
                                  left_on="customer_uuid", right_on="uuid", suffixes=("", "_customer"))
        # keep rows whose customer is missing from customer.csv, and make them easy to spot
        payments["customer"] = payments["customer"].fillna(self.MISSING_CUSTOMER)
        return payments

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