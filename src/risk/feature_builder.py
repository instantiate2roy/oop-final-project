from datetime import date

import numpy as np
import pandas as pd

from ..exceptions import ModelError


class FeatureBuilder:
    """
    Turns the SACCO's history into one row of features per loan per month, ready for a risk model.

    Training rows come from the monthly loan snapshots: each open loan, at the end of each month,
    with what was known on that date. The label says whether the loan went on to reach the
    default DPD (90 by default) within the following months (6 by default). Only snapshots old
    enough for that outcome to be known are used.

    current_features() builds the same features for today's open loans, so the model can score them.
    """

    NUMERIC = ["dpd", "mia", "outstanding_ratio", "months_on_book", "dti", "monthly_income",
               "savings_months_12", "savings_to_income", "guarantor_count", "security_cover",
               "worst_dpd_to_date", "arrears_share_to_date", "broken_ptps_to_date", "age"]
    CATEGORICAL = ["product", "employment_type_id", "repayment_frequency"]

    def __init__(self, data: dict, kb: dict, as_of: date, default_dpd: int = 90, outcome_months: int = 6):
        needed = ["customer", "case", "case_snapshot", "savings_history", "security", "ptp"]
        missing = [name for name in needed if name not in data]
        if missing:
            raise ModelError(f"FeatureBuilder needs these data files: {', '.join(missing)}")
        self.as_of = pd.Timestamp(as_of)
        self.default_dpd = default_dpd
        self.outcome_months = outcome_months

        rows = lambda name: pd.DataFrame(data[name]["rows"])
        self.customers = rows("customer")
        self.cases = rows("case")
        self.snapshots = rows("case_snapshot")
        self.savings = rows("savings_history")
        self.security = rows("security")
        self.ptps = rows("ptp")
        self.products = pd.DataFrame(kb["loan_types"]["rows"])[["uuid", "code"]].rename(columns={"code": "product"})
        self._prepare()

    @property
    def features(self) -> list[str]:
        return self.NUMERIC + self.CATEGORICAL

    # ---- public ---------------------------------------------------------------

    def training_set(self) -> pd.DataFrame:
        """One row per open loan per month-end, with features and the 'defaulted' label."""
        snaps = self.snapshots.sort_values(["case_uuid", "snapshot_date"]).copy()
        # worst DPD over the following outcome_months snapshots of the same loan
        future = snaps.groupby("case_uuid")["dpd"].transform(
            lambda d: d[::-1].rolling(self.outcome_months, min_periods=1).max()[::-1].shift(-1))
        snaps["future_worst_dpd"] = future.fillna(0)

        known_until = self.as_of - pd.DateOffset(months=self.outcome_months)
        points = snaps[(snaps["account_status"] == "OPEN") & (snaps["dpd"] < self.default_dpd)
                       & (snaps["snapshot_date"] <= known_until)]
        points = points.rename(columns={"snapshot_date": "observed_on"})
        frame = self._features_at(points)
        frame["defaulted"] = (frame["future_worst_dpd"] >= self.default_dpd).astype(int)
        return frame

    def portfolio_features(self) -> pd.DataFrame:
        """Features for every open loan at every month-end, plus today's open loans, for reporting."""
        snaps = self.snapshots[self.snapshots["account_status"] == "OPEN"].rename(columns={"snapshot_date": "observed_on"})
        history = self._features_at(snaps[["case_uuid", "observed_on", "dpd", "mia", "arrears_amount",
                                           "outstanding_balance"]])
        today = self.current_features()
        return pd.concat([history[history["observed_on"] < self.as_of], today], ignore_index=True)

    def current_features(self) -> pd.DataFrame:
        """Features for every loan open on the business date, using today's DPD and balances."""
        open_cases = self.cases[self.cases["account_status"] == "OPEN"].copy()
        points = open_cases[["uuid", "dpd", "mia", "arrears_amount", "outstanding_balance"]].rename(
            columns={"uuid": "case_uuid"})
        for column in ["dpd", "mia", "arrears_amount", "outstanding_balance"]:
            points[column] = pd.to_numeric(points[column])
        points["observed_on"] = self.as_of
        return self._features_at(points)

    # ---- building features ----------------------------------------------------

    def _prepare(self) -> None:
        for column in ["loan_amount", "instalment_amount", "total_payable", "term_months"]:
            self.cases[column] = pd.to_numeric(self.cases[column])
        self.cases["disbursement_date"] = self._as_ns(self.cases["disbursement_date"])
        self.cases = self.cases.merge(self.products, left_on="loan_type_uuid", right_on="uuid",
                                      suffixes=("", "_product")).drop(columns="uuid_product")

        self.customers["monthly_income"] = pd.to_numeric(self.customers["monthly_income"])
        self.customers["dob"] = self._as_ns(self.customers["dob"])

        for column in ["dpd", "mia", "arrears_amount", "outstanding_balance"]:
            self.snapshots[column] = pd.to_numeric(self.snapshots[column])
        self.snapshots["snapshot_date"] = self._as_ns(self.snapshots["snapshot_date"])

        self.savings["transaction_date"] = self._as_ns(self.savings["transaction_date"])
        self.savings["balance_after"] = pd.to_numeric(self.savings["balance_after"])
        self.security["value"] = pd.to_numeric(self.security["value"])
        self.ptps["promised_date"] = self._as_ns(self.ptps["promised_date"])

    @staticmethod
    def _as_ns(values: pd.Series) -> pd.Series:
        """Dates in one consistent resolution, so date columns from different files can be merged."""
        return pd.to_datetime(values).astype("datetime64[ns]")

    def _features_at(self, points: pd.DataFrame) -> pd.DataFrame:
        """Features for each (case, observed_on) point, using only what was known on that date."""
        loans = self.cases[["uuid", "customer_uuid", "product", "repayment_frequency", "loan_amount",
                            "instalment_amount", "total_payable", "term_months", "disbursement_date"]]
        points = points.assign(observed_on=self._as_ns(points["observed_on"]))
        frame = points.merge(loans, left_on="case_uuid", right_on="uuid").drop(columns="uuid")
        frame = frame.merge(self.customers[["uuid", "employment_type_id", "monthly_income", "dob"]],
                            left_on="customer_uuid", right_on="uuid").drop(columns="uuid")

        # the loan itself
        frame["outstanding_ratio"] = frame["outstanding_balance"] / frame["total_payable"]
        frame["months_on_book"] = ((frame["observed_on"] - frame["disbursement_date"]).dt.days / 30.44).round(1)
        frame["age"] = ((frame["observed_on"] - frame["dob"]).dt.days / 365.25).astype(int)

        # debt-to-income: the loan's monthly repayment burden against monthly income
        frame["dti"] = (frame["total_payable"] / frame["term_months"]) / frame["monthly_income"]

        frame = self._add_savings(frame)
        frame = self._add_security(frame)
        frame = self._add_repayment_history(frame)
        return frame

    def _add_savings(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Savings habits: months with a deposit in the last 12, and the balance against income."""
        deposits = self.savings[self.savings["transaction_type_id"] == "DEPOSIT"]
        deposit_months = deposits.assign(month=deposits["transaction_date"].dt.to_period("M"))[
            ["customer_uuid", "month"]].drop_duplicates()

        observed_month = frame["observed_on"].dt.to_period("M")
        counts = []
        by_customer = deposit_months.groupby("customer_uuid")["month"].apply(lambda m: set(m))
        for customer, month in zip(frame["customer_uuid"], observed_month):
            months = by_customer.get(customer, set())
            counts.append(sum((month - k) in months for k in range(12)))
        frame["savings_months_12"] = counts

        balances = self.savings.sort_values("transaction_date")[["customer_uuid", "transaction_date", "balance_after"]]
        frame = frame.sort_values("observed_on")
        frame = pd.merge_asof(frame, balances, left_on="observed_on", right_on="transaction_date",
                              by="customer_uuid", direction="backward").drop(columns="transaction_date")
        frame["savings_to_income"] = frame["balance_after"].fillna(0) / frame["monthly_income"]
        return frame.drop(columns="balance_after")

    def _add_security(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Guarantor strength: how many guarantors, and all security as a share of the loan."""
        guarantors = self.security[self.security["security_type_id"] == "GUARANTOR"].groupby("case_uuid").size()
        cover = self.security.groupby("case_uuid")["value"].sum()
        frame["guarantor_count"] = frame["case_uuid"].map(guarantors).fillna(0).astype(int)
        frame["security_cover"] = frame["case_uuid"].map(cover).fillna(0) / frame["loan_amount"]
        return frame

    def _add_repayment_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Worst DPD on any of the customer's loans so far, share of months in arrears, broken promises."""
        snaps = self.snapshots.merge(self.cases[["uuid", "customer_uuid"]], left_on="case_uuid", right_on="uuid")
        snaps = snaps.sort_values("snapshot_date")

        worst = snaps.assign(worst=snaps.groupby("customer_uuid")["dpd"].cummax())[
            ["customer_uuid", "snapshot_date", "worst"]]
        frame = frame.sort_values("observed_on")
        frame = pd.merge_asof(frame, worst, left_on="observed_on", right_on="snapshot_date",
                              by="customer_uuid", direction="backward").drop(columns="snapshot_date")
        frame["worst_dpd_to_date"] = np.maximum(frame["worst"].fillna(0), frame["dpd"])

        share = snaps.assign(late=(snaps["dpd"] > 0).astype(int))
        share["share"] = share.groupby("case_uuid")["late"].transform(lambda s: s.expanding().mean())
        frame = pd.merge_asof(frame.sort_values("observed_on"),
                              share[["case_uuid", "snapshot_date", "share"]].sort_values("snapshot_date"),
                              left_on="observed_on", right_on="snapshot_date", by="case_uuid",
                              direction="backward").drop(columns="snapshot_date")
        frame["arrears_share_to_date"] = frame["share"].fillna(0)

        broken = self.ptps[self.ptps["status"] == "BROKEN"].merge(
            self.cases[["uuid", "customer_uuid"]], left_on="case_uuid", right_on="uuid")
        broken = broken.sort_values("promised_date")
        broken["count"] = broken.groupby("customer_uuid").cumcount() + 1
        frame = pd.merge_asof(frame.sort_values("observed_on"), broken[["customer_uuid", "promised_date", "count"]],
                              left_on="observed_on", right_on="promised_date", by="customer_uuid",
                              direction="backward").drop(columns="promised_date")
        frame["broken_ptps_to_date"] = frame["count"].fillna(0).astype(int)
        return frame.drop(columns=["worst", "share", "count"]).reset_index(drop=True)
