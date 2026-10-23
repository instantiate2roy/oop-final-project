from datetime import date

import pandas as pd

from ..exceptions import DisplayError
from .filters import ChoiceFilter, RangeFilter, TextFilter
from .history_list import HistoryList


class TreatmentHistory(HistoryList):
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
    noun = "treatments"

    def __init__(self, treatment_rows: list[dict], case_rows: list[dict], customer_rows: list[dict],
                 list_page_size: int = 50, filter_cols_per_row: int = 3,
                 treatment_type_rows: list[dict] | None = None, as_of: date | None = None):
        self.treatment_rows = treatment_rows
        # optional lookup of treatment names, e.g. SMS -> "SMS reminders"; without it the code is shown
        self.treatment_type_rows = treatment_type_rows or []
        self.as_of = as_of or date.today()
        super().__init__(case_rows, customer_rows, list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        treatments = pd.DataFrame(self.treatment_rows)
        treatments["dpd_at_start"] = pd.to_numeric(treatments["dpd_at_start"]).astype(int)
        treatments["arrears_at_start"] = pd.to_numeric(treatments["arrears_at_start"]).astype(int)

        start = pd.to_datetime(treatments["start_date"])
        end = pd.to_datetime(treatments["end_date"].replace("", None))
        ends_too_early = end < start
        if ends_too_early.any():
            first = treatments.loc[ends_too_early, "uuid"].iloc[0]
            raise DisplayError(f"{ends_too_early.sum()} treatments end before they start, e.g. {first}")

        # still-running treatments are counted up to the business date
        treatments["days"] = (end.fillna(pd.Timestamp(self.as_of)) - start).dt.days.astype(int)
        treatments["end_date"] = treatments["end_date"].replace("", "running")
        treatments["assigned_to"] = treatments["assigned_to"].replace("", "unassigned")
        treatments["month"] = start.dt.strftime("%Y-%m")

        treatments["treatment"] = treatments["treatment_type_id"]
        if self.treatment_type_rows:
            names = {row["type_id"]: row["name"] for row in self.treatment_type_rows}
            treatments["treatment"] = treatments["treatment_type_id"].map(names).fillna(treatments["treatment_type_id"])
        return self.add_account_and_customer(treatments)

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
