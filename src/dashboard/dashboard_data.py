from dataclasses import dataclass, field

import pandas as pd

from ..risk.roll_rate_model import RollRateModel

ALL = "All"


@dataclass
class DashboardView:
    """Everything the dashboard shows for one choice of filters."""
    business_date: pd.Timestamp
    period_start: pd.Timestamp | None
    filters: dict
    summary: dict
    par_trend: pd.DataFrame
    profiles: pd.DataFrame
    buckets: pd.DataFrame
    treatments: pd.DataFrame
    outcomes: pd.DataFrame
    reachability: pd.DataFrame
    compliance: pd.DataFrame
    payments: pd.DataFrame
    promises: pd.DataFrame
    loans: pd.DataFrame
    tiers: pd.DataFrame
    projection: pd.DataFrame          # balance by state, month 0 to 12; empty when not projected
    write_offs: pd.DataFrame          # projected new write-offs at 3, 6 and 12 months
    notes: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        """The view's tables by name, for exporting."""
        return {"summary": pd.DataFrame([self.summary]), "par_trend": self.par_trend, "profiles": self.profiles,
                "buckets": self.buckets, "treatments": self.treatments, "outcomes": self.outcomes,
                "reachability": self.reachability, "compliance_blocks": self.compliance,
                "payments": self.payments, "promises": self.promises, "loans": self.loans, "tiers": self.tiers,
                "projection": self.projection.reset_index(), "projected_write_offs": self.write_offs}


class DashboardData:
    """
    Works out every figure on the dashboard from the profiled portfolio and the activity files.

    Business dates are the month-ends in the snapshots plus the latest business day. Activity
    (treatments, contacts, payments, promises) for a business date covers the period since the
    previous business date.

    With a fitted RollRateModel, the latest business date also shows where the selected loans'
    balance is projected to be over the next 12 months.
    """

    def __init__(self, portfolio: pd.DataFrame, data: dict, compliance: pd.DataFrame | None = None,
                 roll_rates: RollRateModel | None = None):
        self.portfolio = portfolio
        self.dates = sorted(portfolio["business_date"].unique())
        self.compliance = compliance if compliance is not None else pd.DataFrame()
        self.roll_rates = roll_rates

        rows = lambda name: pd.DataFrame(data[name]["rows"])
        self.treatments = rows("treatment_history").assign(when=lambda t: pd.to_datetime(t["start_date"]))
        self.contacts = rows("contact_history").assign(when=lambda c: pd.to_datetime(c["contact_datetime"]))
        self.payments = rows("payment_history").assign(when=lambda p: pd.to_datetime(p["payment_date"]),
                                                       amount=lambda p: pd.to_numeric(p["amount"]))
        self.promises = rows("ptp").assign(when=lambda p: pd.to_datetime(p["promised_date"]),
                                           promised_amount=lambda p: pd.to_numeric(p["promised_amount"]),
                                           paid_amount=lambda p: pd.to_numeric(p["paid_amount"]))

    def options(self) -> dict[str, list]:
        """Choices for each filter dropdown."""
        return {
            "business_date": [pd.Timestamp(d) for d in self.dates],
            "assigned_to": [ALL, *sorted(self.portfolio["assigned_to"].unique())],
            "profile": [ALL, *sorted(self.portfolio["profile"].unique())],
            "bucket": [ALL, *self._bucket_order(self.portfolio["bucket"].unique())],
            "tier": [ALL, *sorted(self.portfolio["tier"].unique())],
        }

    @staticmethod
    def _bucket_order(buckets) -> list[str]:
        order = ["current", "1-30", "31-60", "61-90", "90+"]
        return sorted(buckets, key=lambda b: order.index(b) if b in order else len(order))

    def _filter(self, frame: pd.DataFrame, assigned_to: str, profile: str, bucket: str, tier: str) -> pd.DataFrame:
        for column, value in (("assigned_to", assigned_to), ("profile", profile), ("bucket", bucket), ("tier", tier)):
            if value != ALL:
                frame = frame[frame[column] == value]
        return frame

    @staticmethod
    def _par(frame: pd.DataFrame, days: int) -> float:
        total = frame["outstanding_balance"].sum()
        return float(frame.loc[frame["dpd"] > days, "outstanding_balance"].sum() / total) if total else 0.0

    def view(self, business_date=None, assigned_to: str = ALL, profile: str = ALL, bucket: str = ALL,
             tier: str = ALL) -> DashboardView:
        day = pd.Timestamp(business_date) if business_date is not None else pd.Timestamp(self.dates[-1])
        position = [pd.Timestamp(d) for d in self.dates].index(day)
        start = pd.Timestamp(self.dates[position - 1]) if position else None

        filtered_all = self._filter(self.portfolio, assigned_to, profile, bucket, tier)
        loans = filtered_all[filtered_all["business_date"] == day]
        loans = loans.assign(expected_loss=loans["pd"] * loans["outstanding_balance"])
        cases = set(loans["case_uuid"])
        in_period = lambda frame: frame[frame["case_uuid"].isin(cases) & (frame["when"] <= day + pd.Timedelta(days=1))
                                        & ((frame["when"] > start) if start is not None else True)]

        trend = filtered_all.groupby("business_date").apply(
            lambda f: pd.Series({"loans": len(f), "outstanding": f["outstanding_balance"].sum(),
                                 "PAR30 %": round(100 * self._par(f, 30), 1), "PAR90 %": round(100 * self._par(f, 90), 1)}),
            include_groups=False).reset_index()

        summary = {"business_date": day.date().isoformat(), "loans": len(loans),
                   "outstanding": int(loans["outstanding_balance"].sum()), "arrears": int(loans["arrears_amount"].sum()),
                   "PAR30 %": round(100 * self._par(loans, 30), 1), "PAR90 %": round(100 * self._par(loans, 90), 1),
                   "average PD %": round(100 * float(loans["pd"].mean()), 1) if len(loans) else 0.0,
                   "expected loss": int(loans["expected_loss"].sum())}

        profiles = loans.groupby("profile").agg(loans=("case_uuid", "size"), balance=("outstanding_balance", "sum"),
                                                average_pd=("pd", "mean")).reset_index()
        profiles["average_pd"] = (100 * profiles["average_pd"]).round(1)
        by_dpd = loans.groupby("bucket").agg(loans_by_dpd=("case_uuid", "size"), balance_by_dpd=("outstanding_balance", "sum"))
        by_mia = loans.groupby("mia_bucket").agg(loans_by_mia=("case_uuid", "size"), balance_by_mia=("outstanding_balance", "sum"))
        buckets = by_dpd.join(by_mia, how="outer").fillna(0).astype(int)
        buckets = buckets.reindex(self._bucket_order(buckets.index)).rename_axis("bucket").reset_index()

        treatments = in_period(self.treatments).groupby("treatment_type_id").size().rename("started").reset_index()
        contacts = in_period(self.contacts)
        outcomes = contacts.groupby(["channel", "outcome"]).size().rename("contacts").reset_index()
        reach = loans.assign(band=pd.cut(loans["reachability"].astype(float), [-0.01, 0.3, 0.6, 1.0],
                                         labels=["low (0-30%)", "medium (31-60%)", "high (61-100%)"]))
        reachability = reach.groupby("band", observed=False).size().rename("loans").reset_index()
        reachability.loc[len(reachability)] = ["no contact attempts", int(loans["reachability"].isna().sum())]

        compliance, notes = pd.DataFrame(columns=["rule", "blocked"]), []
        if position == len(self.dates) - 1 and not self.compliance.empty:
            run = self.compliance[self.compliance["case_uuid"].isin(cases)]
            blocked = run[~run["allowed"]].groupby("blocked_by").size().rename("blocked")
            compliance = blocked.rename_axis("rule").reset_index()
            notes.append(f"Compliance: {len(run)} treatments proposed for the next start-of-day run, "
                         f"{int(run['allowed'].sum())} allowed, {int((~run['allowed']).sum())} blocked.")
        else:
            notes.append("Compliance checks are only recorded for the latest start-of-day run.")

        paid = in_period(self.payments)
        paid = paid[paid["status"] == "POSTED"]
        payments = paid.groupby("channel").agg(payments=("amount", "size"), amount=("amount", "sum")).reset_index()
        promises_due = in_period(self.promises)
        promises = promises_due.groupby("status").agg(promises=("uuid", "size"), promised=("promised_amount", "sum"),
                                                      paid=("paid_amount", "sum")).reset_index()
        settled = promises_due[promises_due["status"].isin(["KEPT", "PARTIALLY_KEPT", "BROKEN"])]
        summary["payments received"] = int(paid["amount"].sum())
        summary["promise keep rate %"] = round(100 * float((settled["status"] == "KEPT").mean()), 1) if len(settled) else 0.0

        tiers = loans.groupby("tier").agg(loans=("case_uuid", "size"), balance=("outstanding_balance", "sum"),
                                          average_pd=("pd", "mean"), expected_loss=("expected_loss", "sum")).reset_index()
        tiers["average_pd"] = (100 * tiers["average_pd"]).round(1)
        tiers["expected_loss"] = tiers["expected_loss"].round().astype(int)

        projection, write_offs = self._projection(loans, latest=position == len(self.dates) - 1, notes=notes)

        detail = loans[["account_no", "customer", "profile", "tier", "bucket", "mia_bucket", "dpd", "mia",
                        "outstanding_balance", "arrears_amount", "pd", "expected_loss", "reachability",
                        "assigned_to"]].sort_values("expected_loss", ascending=False)
        detail["expected_loss"] = detail["expected_loss"].round().astype(int)
        filters = {"assigned_to": assigned_to, "profile": profile, "bucket": bucket, "tier": tier}
        return DashboardView(day, start, filters, summary, trend, profiles, buckets, treatments, outcomes,
                             reachability, compliance, payments, promises, detail.reset_index(drop=True), tiers,
                             projection, write_offs, notes)

    def _projection(self, loans: pd.DataFrame, latest: bool, notes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The roll-rate projection of the selected loans, for the latest business date only."""
        empty = pd.DataFrame(), pd.DataFrame(columns=["months ahead", "written off (UGX)", "share of open balance %"])
        if self.roll_rates is None:
            return empty
        if not latest:
            # the roll rates are learnt from every month, including later ones, so projecting from an
            # earlier date would use what happened afterwards
            notes.append("The roll-rate projection is only shown for the latest business date.")
            return empty
        if loans.empty:
            return empty
        start = self.roll_rates.balances(loans)
        return self.roll_rates.project(start, 12), self.roll_rates.expected_write_offs(start)
