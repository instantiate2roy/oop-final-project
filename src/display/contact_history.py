import pandas as pd

from ..exceptions import DisplayError
from .display import Display
from .filters import ChoiceFilter, TextFilter


class ContactHistory(Display):
    """Every SMS, email and call made to customers, newest first, with the account and customer."""

    title = "Contact history"
    columns = ["contact_datetime", "account_no", "customer", "channel", "contact", "outcome", "notes"]
    sort_by = ["contact_datetime", "account_no"]
    sort_ascending = [False, True]
    MISSING_CUSTOMER = "(customer not found)"

    def __init__(self, contact_history_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3,
                 contact_rows: list[dict] | None = None):
        self.contact_history_rows = contact_history_rows
        self.case_rows = case_rows
        self.customer_rows = customer_rows
        # contacts are optional; without them, the phone number or email column is left blank
        self.contact_rows = contact_rows or []
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        history = pd.DataFrame(self.contact_history_rows)
        history["month"] = pd.to_datetime(history["contact_datetime"]).dt.strftime("%Y-%m")

        # account number and customer name, through the case
        cases = pd.DataFrame(self.case_rows)[["uuid", "account_no", "customer_uuid"]]
        history = history.merge(cases, how="left", left_on="case_uuid", right_on="uuid", suffixes=("", "_case"))
        if history["account_no"].isna().any():
            raise DisplayError(f"{history['account_no'].isna().sum()} contacts refer to cases that don't exist")

        customers = pd.DataFrame(self.customer_rows)
        names = (customers["surname"] + " " + customers["given_name"]).rename("customer")
        history = history.merge(pd.concat([customers["uuid"], names], axis=1), how="left",
                                left_on="customer_uuid", right_on="uuid", suffixes=("", "_customer"))
        # keep rows whose customer is missing from customer.csv, and make them easy to spot
        history["customer"] = history["customer"].fillna(self.MISSING_CUSTOMER)

        # the phone number or email the contact was made on
        if self.contact_rows:
            contacts = pd.DataFrame(self.contact_rows)[["uuid", "value"]].rename(columns={"value": "contact"})
            history = history.merge(contacts, how="left", left_on="contact_uuid",
                                    right_on="uuid", suffixes=("", "_contact"))
            history["contact"] = history["contact"].fillna("")
        else:
            history["contact"] = ""
        return history

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("month", "Month", width="400px"),
            ChoiceFilter("channel", "Channel", width="400px"),
            ChoiceFilter("outcome", "Outcome", width="400px"),
            TextFilter("notes", "Notes", width="400px"),
        ]