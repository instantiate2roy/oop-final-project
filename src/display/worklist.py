import pandas as pd

from .display import Display
from .filters import ChoiceFilter, TextFilter


class Worklist(Display):
    """
    The overdue loans for loan officers to work, ranked by expected loss: PD times outstanding balance.

    Expected loss here assumes nothing is recovered once a loan defaults (a loss given default of 100%),
    so it is an upper bound for putting loans in order, not a provisioning figure.
    """

    title = "Collections worklist"
    columns = ["rank", "account_no", "customer", "bucket", "tier", "profile", "dpd", "outstanding_balance",
               "pd_percent", "expected_loss", "assigned_to", "current_treatment"]
    sort_by = ["rank"]

    def __init__(self, portfolio: pd.DataFrame, business_date=None, list_page_size: int = 50,
                 filter_cols_per_row: int = 3):
        """portfolio: Profiler.portfolio(); business_date defaults to the latest one in it."""
        self.portfolio = portfolio
        self.business_date = (pd.Timestamp(business_date) if business_date is not None
                              else portfolio["business_date"].max())
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        loans = self.portfolio[(self.portfolio["business_date"] == self.business_date)
                               & (self.portfolio["dpd"] > 0)].copy()
        loans["outstanding_balance"] = loans["outstanding_balance"].astype(int)
        loans["expected_loss"] = (loans["pd"] * loans["outstanding_balance"]).round().astype(int)
        loans["pd_percent"] = (100 * loans["pd"]).round(1)
        loans["current_treatment"] = loans["current_treatment"].replace("", "-")
        # the rank stays the loan's place in the whole worklist, whatever the filters show
        loans = loans.sort_values(["expected_loss", "account_no"], ascending=[False, True])
        loans["rank"] = range(1, len(loans) + 1)
        return loans

    def build_filters(self):
        return [
            TextFilter("account_no", "Account no", width="400px"),
            TextFilter("customer", "Customer", width="400px"),
            ChoiceFilter("assigned_to", "Officer", width="400px"),
            ChoiceFilter("bucket", "Bucket", width="400px"),
            ChoiceFilter("tier", "Risk tier", width="400px"),
            ChoiceFilter("profile", "Profile", width="400px"),
        ]
