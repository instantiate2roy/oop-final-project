import bisect

import numpy as np
import pandas as pd

from .customers import add_customer_names
from .policy import Policy
from .risk.feature_builder import FeatureBuilder
from .risk.risk_model import RiskModel

# contact outcomes that mean the customer was reached
REACHED = {"SMS_DELIVERED", "EMAIL_SENT", "RIGHT_PARTY_CONTACT", "PROMISED_TO_PAY", "REFUSED_TO_PAY", "CALL_BACK"}


class Profiler:
    """
    Puts every open loan into a bucket, a risk tier and a profile, using the policy's rules.

    The facts the rules test come from three places: the loan itself (DPD, MIA, balances),
    the risk model (PD), and the contact history (reachability, broken promises).
    """

    def __init__(self, policy: Policy, model: RiskModel, builder: FeatureBuilder, data: dict,
                 reachability_days: int = 90):
        self.policy = policy
        self.model = model
        self.builder = builder
        self.reachability_days = reachability_days

        cases = pd.DataFrame(data["case"]["rows"])[["uuid", "customer_uuid", "account_no"]]
        self.names = add_customer_names(cases, data["customer"]["rows"])[["uuid", "account_no", "customer"]]

        contacts = pd.DataFrame(data["contact_history"]["rows"]).merge(
            cases[["uuid", "customer_uuid"]], left_on="case_uuid", right_on="uuid", suffixes=("", "_case"))
        contacts["when"] = pd.to_datetime(contacts["contact_datetime"]).astype("datetime64[ns]")
        contacts["reached"] = contacts["outcome"].isin(REACHED).astype(int)
        contacts = contacts.sort_values("when")
        self._contacts = {customer: (rows["when"].values, np.cumsum(rows["reached"].values))
                          for customer, rows in contacts.groupby("customer_uuid")}

        treatments = pd.DataFrame(data["treatment_history"]["rows"])
        treatments["start"] = pd.to_datetime(treatments["start_date"]).astype("datetime64[ns]")
        treatments["end"] = pd.to_datetime(treatments["end_date"].replace("", None)).astype("datetime64[ns]")
        self._treatments = treatments.sort_values("start")

    def mia_bucket(self, mia: float) -> str:
        """Bucket by months in arrears: each month behind counts as 30 days."""
        return self.policy.bucket_for(int(np.ceil(round(mia * 30, 6))))

    def portfolio(self) -> pd.DataFrame:
        """One row per open loan per business date (month-ends plus today), with bucket, PD, tier and profile."""
        frame = self.builder.portfolio_features()
        frame["pd"] = self.model.predict_pd(frame).values
        frame["tier"] = frame["pd"].map(self.policy.tier_for)
        frame["bucket"] = frame["dpd"].astype(int).map(self.policy.bucket_for)
        frame["mia_bucket"] = frame["mia"].map(self.mia_bucket)
        frame["reachability"] = [self._reachability(c, d) for c, d in zip(frame["customer_uuid"], frame["observed_on"])]
        frame["profile"] = [self.policy.profile_for(self._facts(row)) for row in frame.to_dict("records")]
        frame = self._add_assignment(frame)
        frame = frame.merge(self.names, left_on="case_uuid", right_on="uuid").drop(columns="uuid")
        return frame.rename(columns={"observed_on": "business_date", "broken_ptps_to_date": "broken_ptps"})

    def _facts(self, row: dict) -> dict:
        return {"bucket": row["bucket"], "dpd": row["dpd"], "mia": row["mia"],
                "outstanding_balance": row["outstanding_balance"], "outstanding_ratio": row["outstanding_ratio"],
                "arrears_amount": row["arrears_amount"], "reachability": row["reachability"],
                "broken_ptps": row["broken_ptps_to_date"], "pd": row["pd"], "tier": row["tier"]}

    def _reachability(self, customer: str, day: pd.Timestamp) -> float | None:
        """Share of contact attempts in the last N days that reached the customer; None if no attempts."""
        if customer not in self._contacts:
            return None
        times, reached = self._contacts[customer]
        end = bisect.bisect_right(times, np.datetime64(day + pd.Timedelta(days=1)))
        start = bisect.bisect_right(times, np.datetime64(day - pd.Timedelta(days=self.reachability_days)))
        attempts = end - start
        if attempts == 0:
            return None
        successes = reached[end - 1] - (reached[start - 1] if start else 0)
        return round(successes / attempts, 2)

    def _add_assignment(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Who was working each loan on each date: the treatment running then, if any."""
        frame = frame.sort_values("observed_on")
        latest = pd.merge_asof(frame[["case_uuid", "observed_on"]].reset_index(),
                               self._treatments[["case_uuid", "start", "end", "assigned_to", "treatment_type_id"]],
                               left_on="observed_on", right_on="start", by="case_uuid", direction="backward")
        running = latest["start"].notna() & (latest["end"].isna() | (latest["end"] >= latest["observed_on"]))
        latest["assigned_to"] = latest["assigned_to"].where(running, "Not in collections")
        latest["current_treatment"] = latest["treatment_type_id"].where(running, "")
        frame["assigned_to"] = latest.set_index("index")["assigned_to"]
        frame["current_treatment"] = latest.set_index("index")["current_treatment"]
        return frame.reset_index(drop=True)
