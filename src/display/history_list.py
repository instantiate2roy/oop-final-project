import pandas as pd

from ..customers import MISSING_CUSTOMER, add_customer_names
from ..exceptions import DisplayError
from .display import Display


class HistoryList(Display):
    """
    A list of things that happened on loan cases (contacts, payments, promises, treatments),
    each shown with the case's account number and the customer's name.

    Subclasses build their own rows in build_frame() and pass them through
    add_account_and_customer(), which every history list needs.
    """

    MISSING_CUSTOMER = MISSING_CUSTOMER
    noun = "rows"                 # what the rows are, for messages, e.g. "contacts"

    def __init__(self, case_rows: list[dict], customer_rows: list[dict], list_page_size: int = 50,
                 filter_cols_per_row: int = 3):
        self.case_rows = case_rows
        self.customer_rows = customer_rows
        super().__init__(list_page_size, filter_cols_per_row)

    def add_account_and_customer(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Add account_no and customer_uuid through the case, then the customer's name.

        A row for a case that doesn't exist is an error in the data; a row whose customer is
        missing is kept and shown as "(customer not found)".
        """
        cases = pd.DataFrame(self.case_rows)[["uuid", "account_no", "customer_uuid"]].rename(
            columns={"uuid": "case_uuid"})
        frame = frame.merge(cases, how="left", on="case_uuid")
        if frame["account_no"].isna().any():
            raise DisplayError(f"{frame['account_no'].isna().sum()} {self.noun} refer to cases that don't exist")
        return add_customer_names(frame, self.customer_rows, self.MISSING_CUSTOMER)
