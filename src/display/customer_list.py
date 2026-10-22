from datetime import date

import pandas as pd

from .display import Display
from .filters import ChoiceFilter, RangeFilter, TextFilter


class CustomerList(Display):
    """
    Searchable list of customers, with age worked out on the business date.

    Each row also shows the customer's main phone, email and home address, picked from
    their contacts and addresses. A customer can have several of each, so the list shows
    the best one, and the number of valid contacts on record.
    """

    title = "Customers"
    columns = ["surname", "given_name", "sex", "age", "nin",
               "phone", "email", "home_address", "district", "contacts"]
    sort_by = ["surname", "given_name"]

    # best address first: a verified address beats an unverified one, and so on
    ADDRESS_STATUS_ORDER = ["VERIFIED", "UNVERIFIED", "MOVED", "NOT_FOUND"]

    def __init__(self, customer_rows: list[dict], as_of: date, list_page_size: int = 50,
                 filter_cols_per_row: int = 3, address_rows: list[dict] | None = None,
                 contact_rows: list[dict] | None = None, location_rows: list[dict] | None = None):
        self.customer_rows = customer_rows
        self.as_of = as_of
        self.address_rows = address_rows or []
        self.contact_rows = contact_rows or []
        self.location_rows = location_rows or []
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.customer_rows)
        dob = pd.to_datetime(frame["dob"])
        had_birthday = (dob.dt.month < self.as_of.month) | (
            (dob.dt.month == self.as_of.month) & (dob.dt.day <= self.as_of.day))
        frame["age"] = self.as_of.year - dob.dt.year - (~had_birthday).astype(int)

        frame = frame.merge(self._main_contacts(), how="left", left_on="uuid", right_index=True)
        frame = frame.merge(self._home_addresses(), how="left", left_on="uuid", right_index=True)
        frame["contacts"] = frame["contacts"].fillna(0).astype(int)
        return frame.fillna({"phone": "", "email": "", "home_address": "", "district": ""})

    def build_filters(self):
        return [
            TextFilter("surname", "Surname", width="400px"),
            TextFilter("given_name", "Given name", width="400px"),
            TextFilter("nin", "NIN", width="400px"),     # 14 characters plus the label
            TextFilter("phone", "Phone", width="400px"),
            RangeFilter("age", "Age", width="400px"),
            ChoiceFilter("district", "District", width="400px"),
            ChoiceFilter("sex", "Sex", width="200px"),
        ]

    def _main_contacts(self) -> pd.DataFrame:
        """For each customer: first valid mobile, first valid email, and how many valid contacts."""
        columns = ["phone", "email", "contacts"]
        if not self.contact_rows:
            return pd.DataFrame(columns=columns)
        contacts = pd.DataFrame(self.contact_rows)
        valid = contacts[contacts["valid"] == "Y"]
        by_customer = valid.groupby("customer_uuid")
        return pd.DataFrame({
            "phone": valid[valid["contact_type_id"] == "MOBILE_PHONE"].groupby("customer_uuid")["value"].first(),
            "email": valid[valid["contact_type_id"] == "EMAIL"].groupby("customer_uuid")["value"].first(),
            "contacts": by_customer.size(),
        })

    def _home_addresses(self) -> pd.DataFrame:
        """For each customer: their best home address and its village and district."""
        columns = ["home_address", "district"]
        if not self.address_rows:
            return pd.DataFrame(columns=columns)
        addresses = pd.DataFrame(self.address_rows)
        homes = addresses[addresses["address_type_id"] == "HOME"].copy()

        if self.location_rows:
            places = pd.DataFrame(self.location_rows)[["uuid", "village", "district"]]
            homes = homes.merge(places, how="left", left_on="location_uuid",
                                right_on="uuid", suffixes=("", "_location"))
            homes["home_address"] = homes["address_line"] + ", " + homes["village"]
        else:
            homes["home_address"] = homes["address_line"]
            homes["district"] = ""

        rank = {status: position for position, status in enumerate(self.ADDRESS_STATUS_ORDER)}
        homes["rank"] = homes["status"].map(rank).fillna(len(rank))
        best = homes.sort_values("rank").groupby("customer_uuid").first()
        return best[columns]