import pandas as pd

from .filters import ChoiceFilter, TextFilter
from .history_list import HistoryList


class ContactHistory(HistoryList):
    """Every SMS, email and call made to customers, newest first, with the account and customer."""

    title = "Contact history"
    columns = ["contact_datetime", "account_no", "customer", "channel", "contact", "outcome", "notes"]
    sort_by = ["contact_datetime", "account_no"]
    sort_ascending = [False, True]
    noun = "contacts"

    def __init__(self, contact_history_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3,
                 contact_rows: list[dict] | None = None):
        self.contact_history_rows = contact_history_rows
        # contacts are optional; without them, the phone number or email column is left blank
        self.contact_rows = contact_rows or []
        super().__init__(case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        history = pd.DataFrame(self.contact_history_rows)
        history["month"] = pd.to_datetime(history["contact_datetime"]).dt.strftime("%Y-%m")
        history = self.add_account_and_customer(history)

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
