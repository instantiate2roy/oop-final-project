import json
from pathlib import Path

import pandas as pd

from .exceptions import PolicyError


class Policy:
    """
    The collections policy from configs/policy.json: buckets, risk tiers, profile rules,
    the treatment matrix and the compliance limits.

    The file is checked when it is loaded, so a mistake in the policy stops the system
    with a clear message instead of quietly treating customers the wrong way. A missing or
    badly formed setting raises PolicyError too, never a bare KeyError.

    The raw settings are private: other classes read the policy through its properties and methods.
    """

    # fields a profile rule may test, and the tests it may use
    PROFILE_FIELDS = {"bucket", "dpd", "mia", "outstanding_balance", "outstanding_ratio",
                      "arrears_amount", "reachability", "broken_ptps", "pd", "tier"}
    OPERATORS = {"in", "min", "max", "equals"}
    WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def __init__(self, data: dict, treatment_types: list[str] | None = None):
        self._data = data
        self.treatment_types = set(treatment_types or [])
        try:
            self._check()
        except (KeyError, TypeError, AttributeError) as error:
            # a safety net: _check names the usual mistakes itself, this catches any shape it didn't expect
            raise PolicyError(f"The policy is badly formed: {type(error).__name__} {error}") from error

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
        return self._data["buckets"]["edges"]

    @property
    def risk_tiers(self) -> list[dict]:
        """Risk tiers from lowest to highest PD; the last has no upper limit."""
        return self._data["risk_tiers"]

    @property
    def profiles(self) -> list[dict]:
        """Profile rules in the order they are checked."""
        return sorted(self._data["profiles"], key=lambda rule: rule["priority"])

    @property
    def default_profile(self) -> str:
        """The profile for anyone no rule matches."""
        return self._data["default_profile"]

    @property
    def fallback_treatments(self) -> dict[str, str]:
        """The treatment to use instead when a customer can't be reached on the first one, e.g. SMS -> EMAIL."""
        return dict(self._data.get("fallback_treatments", {}))

    @property
    def treatment_matrix(self) -> dict[str, dict[str, str]]:
        return self._data["treatment_matrix"]

    @property
    def compliance(self) -> dict:
        return self._data["compliance"]

    # ---- using the policy -----------------------------------------------------

    def bucket_for(self, dpd: int) -> str:
        """The bucket a loan falls in for its days past due."""
        if dpd < 0:
            raise PolicyError(f"DPD can't be negative: {dpd}")
        for bucket in self.buckets:
            if dpd >= bucket["lower"] and (bucket["upper"] is None or dpd <= bucket["upper"]):
                return bucket["name"]
        raise PolicyError(f"No bucket covers a DPD of {dpd}")

    def tier_for(self, pd_value: float) -> str:
        """The risk tier for a probability of default: the first tier whose max_pd is above it."""
        if not 0 <= pd_value <= 1:
            raise PolicyError(f"PD must be between 0 and 1, not {pd_value}")
        for tier in self.risk_tiers:
            if tier["max_pd"] is None or pd_value < tier["max_pd"]:
                return tier["name"]
        raise PolicyError(f"No risk tier covers a PD of {pd_value}")

    def profile_for(self, facts: dict) -> str:
        """The first profile, by priority, whose conditions all match; otherwise the default profile."""
        for rule in self.profiles:
            if all(self._condition_matches(facts.get(field), test)
                   for field, test in rule["conditions"].items()):
                return rule["name"]
        return self._data["default_profile"]

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
        measure = self._data["buckets"].get("measure", "dpd").upper()
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
            "Conditions (all must match)": self.describe(rule["conditions"]),
        } for rule in self.profiles]
        rows.append({"Priority": "default", "Profile": self._data["default_profile"],
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
             "Setting": ", ".join(f"{a} → {b}" for a, b in self._data.get("fallback_treatments", {}).items())},
        ])
        return channels, general

    @staticmethod
    def describe(conditions: dict) -> str:
        """A profile rule's conditions in words, e.g. 'bucket in 61-90, 90+ and reachability ≤ 0.3'."""
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
        if not isinstance(self._data, dict):
            raise PolicyError(f"The policy must be a JSON object, not {type(self._data).__name__}")
        for section in ["buckets", "risk_tiers", "profiles", "default_profile", "treatment_matrix", "compliance"]:
            if section not in self._data:
                raise PolicyError(f"Policy is missing the '{section}' section")
        self._check_shape()
        self._check_buckets()
        self._check_tiers()
        self._check_profiles()
        self._check_matrix()
        self._check_compliance()

    @staticmethod
    def _require(item, keys: list[str], where: str) -> None:
        """Stop with a clear message when a part of the policy isn't an object or lacks a setting."""
        if not isinstance(item, dict):
            raise PolicyError(f"{where} must be an object, not {type(item).__name__}")
        missing = [key for key in keys if key not in item]
        if missing:
            raise PolicyError(f"{where} is missing {', '.join(repr(key) for key in missing)}")

    @staticmethod
    def _require_list(item, where: str) -> None:
        if not isinstance(item, list):
            raise PolicyError(f"{where} must be a list, not {type(item).__name__}")

    def _check_shape(self) -> None:
        """Every setting the rest of the policy checks, and the system, relies on is present."""
        self._require(self._data["buckets"], ["edges"], "buckets")
        self._require_list(self.buckets, "buckets.edges")
        for number, bucket in enumerate(self.buckets, start=1):
            self._require(bucket, ["name", "lower", "upper"], f"Bucket {number}")

        self._require_list(self.risk_tiers, "risk_tiers")
        for number, tier in enumerate(self.risk_tiers, start=1):
            self._require(tier, ["name", "max_pd"], f"Risk tier {number}")

        self._require_list(self._data["profiles"], "profiles")
        for number, rule in enumerate(self._data["profiles"], start=1):
            self._require(rule, ["name", "priority", "conditions"], f"Profile rule {number}")
            self._require(rule["conditions"], [], f"The conditions of profile '{rule['name']}'")
            for field, test in rule["conditions"].items():
                self._require(test, [], f"Profile '{rule['name']}', condition on '{field}'")

        self._require(self.treatment_matrix, [], "treatment_matrix")
        for profile, row in self.treatment_matrix.items():
            self._require(row, [], f"Treatment matrix row '{profile}'")
        self._require(self._data.get("fallback_treatments", {}), [], "fallback_treatments")

        self._require(self.compliance, ["contact_hours", "max_contacts_per_day", "hold_while_ptp_pending",
                                        "do_not_treat_case_statuses", "channels"], "compliance")
        self._require(self.compliance["contact_hours"], ["start", "end", "days"], "compliance.contact_hours")
        self._require(self.compliance["channels"], [], "compliance.channels")
        for channel, limits in self.compliance["channels"].items():
            self._require(limits, ["max_per_day", "max_per_week", "min_gap_hours"], f"Channel limits for {channel}")

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

    def _check_tiers(self) -> None:
        tiers = self.risk_tiers
        if not tiers:
            raise PolicyError("At least one risk tier is needed")
        names = [tier["name"] for tier in tiers]
        if len(set(names)) != len(names):
            raise PolicyError("Two risk tiers have the same name")
        limits = [tier["max_pd"] for tier in tiers[:-1]]
        if None in limits:
            raise PolicyError("Only the last risk tier can be open-ended")
        if tiers[-1]["max_pd"] is not None:
            raise PolicyError(f"The last risk tier, '{tiers[-1]['name']}', must have no max_pd")
        if any(not 0 < limit <= 1 for limit in limits) or limits != sorted(set(limits)):
            raise PolicyError("Risk tier max_pd values must rise from tier to tier and lie between 0 and 1")

    def _check_profiles(self) -> None:
        bucket_names = {b["name"] for b in self.buckets}
        names = [rule["name"] for rule in self._data["profiles"]]
        priorities = [rule["priority"] for rule in self._data["profiles"]]
        if len(set(names)) != len(names):
            raise PolicyError("Two profile rules have the same name")
        if len(set(priorities)) != len(priorities):
            raise PolicyError("Two profile rules have the same priority")
        for rule in self._data["profiles"]:
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
        profile_names = {rule["name"] for rule in self._data["profiles"]} | {self._data["default_profile"]}
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
