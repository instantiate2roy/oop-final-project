import json
from pathlib import Path

import pandas as pd

from .exceptions import PolicyError


class Policy:
    """
    The collections policy from configs/policy.json: buckets, profile rules,
    the treatment matrix and the compliance limits.

    The file is checked when it is loaded, so a mistake in the policy stops the system
    with a clear message instead of quietly treating customers the wrong way.
    """

    # fields a profile rule may test, and the tests it may use
    PROFILE_FIELDS = {"bucket", "dpd", "mia", "outstanding_balance", "outstanding_ratio",
                      "arrears_amount", "reachability", "broken_ptps", "pd"}
    OPERATORS = {"in", "min", "max", "equals"}
    WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def __init__(self, data: dict, treatment_types: list[str] | None = None):
        self.data = data
        self.treatment_types = set(treatment_types or [])
        self._check()

    @classmethod
    def load(cls, path: str | Path = "configs/policy.json",
             treatment_type_rows: list[dict] | None = None) -> "Policy":
        """Read and check the policy file. Pass kb['treatment_type']['rows'] to check treatment codes too."""
        path = Path(path)
        if not path.is_file():
            raise PolicyError(f"Policy file not found: {path.resolve()}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise PolicyError(f"{path.name} is not valid JSON: {error}") from error
        types = [row["type_id"] for row in treatment_type_rows] if treatment_type_rows else None
        return cls(data, types)

    # ---- the policy's parts ---------------------------------------------------

    @property
    def buckets(self) -> list[dict]:
        return self.data["buckets"]["edges"]

    @property
    def profiles(self) -> list[dict]:
        """Profile rules in the order they are checked."""
        return sorted(self.data["profiles"], key=lambda rule: rule["priority"])

    @property
    def treatment_matrix(self) -> dict[str, dict[str, str]]:
        return self.data["treatment_matrix"]

    @property
    def compliance(self) -> dict:
        return self.data["compliance"]

    # ---- using the policy -----------------------------------------------------

    def bucket_for(self, dpd: int) -> str:
        """The bucket a loan falls in for its days past due."""
        if dpd < 0:
            raise PolicyError(f"DPD can't be negative: {dpd}")
        for bucket in self.buckets:
            if dpd >= bucket["lower"] and (bucket["upper"] is None or dpd <= bucket["upper"]):
                return bucket["name"]
        raise PolicyError(f"No bucket covers a DPD of {dpd}")

    def profile_for(self, facts: dict) -> str:
        """The first profile, by priority, whose conditions all match; otherwise the default profile."""
        for rule in self.profiles:
            if all(self._condition_matches(facts.get(field), test)
                   for field, test in rule["conditions"].items()):
                return rule["name"]
        return self.data["default_profile"]

    def treatment_for(self, profile: str, bucket: str) -> str | None:
        """The treatment for a profile and bucket, or None when the matrix has none."""
        return self.treatment_matrix.get(profile, {}).get(bucket)

    @staticmethod
    def _condition_matches(value, test: dict) -> bool:
        if value is None:
            return False
        if "in" in test and value not in test["in"]:
            return False
        if "equals" in test and value != test["equals"]:
            return False
        if "min" in test and value < test["min"]:
            return False
        if "max" in test and value > test["max"]:
            return False
        return True

    # ---- tables for the notebook ----------------------------------------------

    def buckets_table(self) -> pd.DataFrame:
        measure = self.data["buckets"]["measure"].upper()
        return pd.DataFrame([{
            "Bucket": b["name"],
            f"From ({measure})": b["lower"],
            f"To ({measure})": "no limit" if b["upper"] is None else b["upper"],
        } for b in self.buckets])

    def profiles_table(self) -> pd.DataFrame:
        rows = [{
            "Priority": rule["priority"],
            "Profile": rule["name"],
            "Description": rule.get("description", ""),
            "Conditions (all must match)": self._describe(rule["conditions"]),
        } for rule in self.profiles]
        rows.append({"Priority": "default", "Profile": self.data["default_profile"],
                     "Description": "Anyone no rule matches", "Conditions (all must match)": "-"})
        return pd.DataFrame(rows)

    def treatment_matrix_table(self) -> pd.DataFrame:
        """Profiles down the side, overdue buckets across the top, treatments in the cells."""
        overdue = [b["name"] for b in self.buckets if b["upper"] != 0]
        table = pd.DataFrame(
            [[self.treatment_for(rule["name"], bucket) or "-" for bucket in overdue] for rule in self.profiles],
            index=[rule["name"] for rule in self.profiles], columns=overdue)
        table.index.name = "Profile"
        return table

    def compliance_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Limits per channel, and the general rules that apply to every contact."""
        channels = pd.DataFrame([{
            "Channel": name,
            "Max per day": limits["max_per_day"],
            "Max per week": limits["max_per_week"],
            "Min gap (hours)": limits["min_gap_hours"],
        } for name, limits in self.compliance["channels"].items()])

        hours = self.compliance["contact_hours"]
        general = pd.DataFrame([
            {"Rule": "Contact hours", "Setting": f"{hours['start']} to {hours['end']}"},
            {"Rule": "Contact days", "Setting": ", ".join(hours["days"])},
            {"Rule": "Max contacts per customer per day (all channels)",
             "Setting": self.compliance["max_contacts_per_day"]},
            {"Rule": "Hold while a promise to pay is pending",
             "Setting": "Yes" if self.compliance["hold_while_ptp_pending"] else "No"},
            {"Rule": "Never treat cases with status",
             "Setting": ", ".join(self.compliance["do_not_treat_case_statuses"])},
            {"Rule": "Fallback treatments",
             "Setting": ", ".join(f"{a} → {b}" for a, b in self.data.get("fallback_treatments", {}).items())},
        ])
        return channels, general

    @staticmethod
    def _describe(conditions: dict) -> str:
        parts = []
        for field, test in conditions.items():
            if "in" in test:
                parts.append(f"{field} in {', '.join(map(str, test['in']))}")
            if "equals" in test:
                parts.append(f"{field} = {test['equals']}")
            if "min" in test:
                parts.append(f"{field} ≥ {test['min']}")
            if "max" in test:
                parts.append(f"{field} ≤ {test['max']}")
        return " and ".join(parts)

    # ---- checks ---------------------------------------------------------------

    def _check(self) -> None:
        for section in ["buckets", "profiles", "default_profile", "treatment_matrix", "compliance"]:
            if section not in self.data:
                raise PolicyError(f"Policy is missing the '{section}' section")
        self._check_buckets()
        self._check_profiles()
        self._check_matrix()
        self._check_compliance()

    def _check_buckets(self) -> None:
        edges = self.buckets
        if not edges or edges[0]["lower"] != 0:
            raise PolicyError("Buckets must start at 0")
        for before, after in zip(edges, edges[1:]):
            if before["upper"] is None:
                raise PolicyError(f"Only the last bucket can be open-ended, but '{before['name']}' is")
            if after["lower"] != before["upper"] + 1:
                raise PolicyError(f"Buckets '{before['name']}' and '{after['name']}' leave a gap or overlap")
        if edges[-1]["upper"] is not None:
            raise PolicyError(f"The last bucket, '{edges[-1]['name']}', must have no upper limit")

    def _check_profiles(self) -> None:
        bucket_names = {b["name"] for b in self.buckets}
        names = [rule["name"] for rule in self.data["profiles"]]
        priorities = [rule["priority"] for rule in self.data["profiles"]]
        if len(set(names)) != len(names):
            raise PolicyError("Two profile rules have the same name")
        if len(set(priorities)) != len(priorities):
            raise PolicyError("Two profile rules have the same priority")
        for rule in self.data["profiles"]:
            for field, test in rule["conditions"].items():
                if field not in self.PROFILE_FIELDS:
                    raise PolicyError(f"Profile '{rule['name']}' uses unknown field '{field}'")
                unknown = set(test) - self.OPERATORS
                if unknown:
                    raise PolicyError(f"Profile '{rule['name']}' uses unknown test {sorted(unknown)} on '{field}'")
                if "min" in test and "max" in test and test["min"] > test["max"]:
                    raise PolicyError(f"Profile '{rule['name']}': min is above max for '{field}'")
                if field == "bucket" and set(test.get("in", [])) - bucket_names:
                    raise PolicyError(f"Profile '{rule['name']}' refers to unknown buckets "
                                      f"{sorted(set(test['in']) - bucket_names)}")

    def _check_matrix(self) -> None:
        profile_names = {rule["name"] for rule in self.data["profiles"]} | {self.data["default_profile"]}
        bucket_names = {b["name"] for b in self.buckets}
        for profile, row in self.treatment_matrix.items():
            if profile not in profile_names:
                raise PolicyError(f"Treatment matrix refers to unknown profile '{profile}'")
            for bucket, treatment in row.items():
                if bucket not in bucket_names:
                    raise PolicyError(f"Treatment matrix row '{profile}' refers to unknown bucket '{bucket}'")
                if self.treatment_types and treatment not in self.treatment_types:
                    raise PolicyError(f"Treatment matrix uses unknown treatment '{treatment}' for {profile}/{bucket}")

    def _check_compliance(self) -> None:
        hours = self.compliance["contact_hours"]
        if hours["start"] >= hours["end"]:
            raise PolicyError(f"Contact hours start ({hours['start']}) must be before they end ({hours['end']})")
        bad_days = set(hours["days"]) - set(self.WEEKDAYS)
        if bad_days:
            raise PolicyError(f"Unknown contact days: {sorted(bad_days)}")
        for channel, limits in self.compliance["channels"].items():
            if limits["max_per_day"] > limits["max_per_week"]:
                raise PolicyError(f"{channel}: max per day can't be more than max per week")