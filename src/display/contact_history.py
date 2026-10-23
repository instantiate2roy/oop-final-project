import pandas as pd

from .case_history import CaseHistory
from .filters import ChoiceFilter, TextFilter


class ContactHistory(CaseHistory):
    """Every SMS, email and call made to customers, newest first."""

    title = "Contact history"
    columns = ["contact_datetime", "account_no", "customer", "channel", "contact", "outcome", "notes"]
    sort_by = ["contact_datetime", "account_no"]
    sort_ascending = [False, True]

    def __init__(self, history_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 contact_rows: list[dict] | None = None, list_page_size: int = 50,
                 filter_cols_per_row: int = 3):
        # contacts are optional; without them, the phone number or email column is left blank
        self.contact_rows = contact_rows or []
        super().__init__(history_rows, case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.contact_rows:
            contacts = pd.DataFrame(self.contact_rows)[["uuid", "value"]].rename(columns={"value": "contact"})
            history = history.merge(contacts, how="left", left_on="contact_uuid",
                                    right_on="uuid", suffixes=("", "_contact"))
        history["contact"] = history.get("contact", pd.Series("", index=history.index)).fillna("")
        return self.add_month(history, "contact_datetime")

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("month", "Month", width="400px"),
            ChoiceFilter("channel", "Channel", width="400px"),
            ChoiceFilter("outcome", "Outcome", width="400px"),
            TextFilter("notes", "Notes", width="400px"),
        ]