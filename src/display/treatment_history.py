from datetime import date

import pandas as pd

from ..exceptions import DisplayError
from .case_history import CaseHistory
from .filters import ChoiceFilter, RangeFilter, TextFilter


class TreatmentHistory(CaseHistory):
    """
    Every collection treatment started on a case, newest first.

    A treatment runs from start_date to end_date; a blank end_date means it is still running.
    'days' is how long it ran, or has been running so far on the business date.
    """

    title = "Treatment history"
    columns = ["start_date", "end_date", "days", "account_no", "customer", "treatment",
               "dpd_at_start", "arrears_at_start", "assigned_to", "status"]
    sort_by = ["start_date", "account_no"]
    sort_ascending = [False, True]

    def __init__(self, history_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 as_of: date | None = None, treatment_type_rows: list[dict] | None = None,
                 list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.as_of = as_of or date.today()
        # optional lookup of treatment names, e.g. SMS -> "SMS reminder"; without it the code is shown
        self.treatment_type_rows = treatment_type_rows or []
        super().__init__(history_rows, case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        history["dpd_at_start"] = pd.to_numeric(history["dpd_at_start"]).astype(int)
        history["arrears_at_start"] = pd.to_numeric(history["arrears_at_start"]).astype(int)

        start = pd.to_datetime(history["start_date"])
        end = pd.to_datetime(history["end_date"].replace("", None))
        finished_before_start = end < start
        if finished_before_start.any():
            first = history.loc[finished_before_start, "uuid"].iloc[0]
            raise DisplayError(f"{finished_before_start.sum()} treatments end before they start, e.g. {first}")

        # still-running treatments are counted up to the business date
        history["days"] = (end.fillna(pd.Timestamp(self.as_of)) - start).dt.days.astype(int)
        history["end_date"] = history["end_date"].replace("", "running")

        history["treatment"] = history["treatment_type_id"]
        if self.treatment_type_rows:
            names = {row["type_id"]: row["name"] for row in self.treatment_type_rows}
            history["treatment"] = history["treatment_type_id"].map(names).fillna(history["treatment_type_id"])

        history["assigned_to"] = history["assigned_to"].replace("", "unassigned")
        return self.add_month(history, "start_date")

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("month", "Started in", width="400px"),
            ChoiceFilter("treatment", "Treatment", width="400px"),
            ChoiceFilter("status", "Status", width="400px"),
            ChoiceFilter("assigned_to", "Assigned to", width="400px"),
            RangeFilter("dpd_at_start", "DPD at start", width="400px"),
            RangeFilter("arrears_at_start", "Arrears", step=10_000, width="400px"),
            RangeFilter("days", "Days", width="400px"),
        ]