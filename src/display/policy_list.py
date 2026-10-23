import pandas as pd

from ..policy import Policy
from .display import Display
from .filters import ChoiceFilter, TextFilter


class PolicyList(Display):
    """
    Searchable list of every rule in the collections policy.

    Each row is one setting: a bucket, a profile rule, a cell of the treatment matrix,
    a channel limit or a general compliance rule. Filter by section to see one part
    of the policy, or search for a profile, bucket or treatment across all of it.
    """

    title = "Collections policy"
    columns = ["section", "item", "setting"]
    sort_by = ["section_order", "order"]

    # the order the sections appear in when the list isn't filtered
    SECTIONS = ["Bucket", "Profile rule", "Treatment", "Channel limit", "General rule"]

    def __init__(self, policy: Policy, list_page_size: int = 50, filter_cols_per_row: int = 3):
        self.policy = policy
        super().__init__(list_page_size, filter_cols_per_row)

    def build_frame(self) -> pd.DataFrame:
        rows = (self._bucket_rows() + self._profile_rows() + self._treatment_rows()
                + self._channel_rows() + self._general_rows())
        frame = pd.DataFrame(rows)
        frame["section_order"] = frame["section"].map(self.SECTIONS.index)
        return frame

    def build_filters(self):
        return [
            ChoiceFilter("section", "Section", width="400px"),
            TextFilter("item", "Item", width="400px"),
            TextFilter("setting", "Setting", width="400px"),
        ]

    def _bucket_rows(self) -> list[dict]:
        measure = self.policy.data["buckets"]["measure"].upper()
        return [{
            "section": "Bucket", "order": position, "item": bucket["name"],
            "setting": f"{measure} {bucket['lower']} and over" if bucket["upper"] is None
                       else f"{measure} {bucket['lower']} to {bucket['upper']}",
        } for position, bucket in enumerate(self.policy.buckets)]

    def _profile_rows(self) -> list[dict]:
        rows = [{
            "section": "Profile rule", "order": rule["priority"],
            "item": f"{rule['priority']}. {rule['name']}",
            "setting": f"{self.policy._describe(rule['conditions'])}  ({rule.get('description', '')})",
        } for rule in self.policy.profiles]
        rows.append({"section": "Profile rule", "order": len(rows) + 1,
                     "item": f"default: {self.policy.data['default_profile']}",
                     "setting": "anyone no rule matches"})
        return rows

    def _treatment_rows(self) -> list[dict]:
        rows = []
        for rule in self.policy.profiles:
            for bucket in self.policy.buckets:
                treatment = self.policy.treatment_for(rule["name"], bucket["name"])
                if treatment:
                    rows.append({"section": "Treatment", "order": len(rows),
                                 "item": f"{rule['name']} in {bucket['name']}", "setting": treatment})
        for original, fallback in self.policy.data.get("fallback_treatments", {}).items():
            rows.append({"section": "Treatment", "order": len(rows),
                         "item": f"fallback for {original}", "setting": f"{fallback}, if {original} can't be used"})
        return rows

    def _channel_rows(self) -> list[dict]:
        return [{
            "section": "Channel limit", "order": position, "item": channel,
            "setting": f"at most {limits['max_per_day']} a day and {limits['max_per_week']} a week, "
                       f"at least {limits['min_gap_hours']} hours apart",
        } for position, (channel, limits) in enumerate(self.policy.compliance["channels"].items())]

    def _general_rows(self) -> list[dict]:
        _, general = self.policy.compliance_tables()
        # fallbacks are already listed with the treatments
        general = general[general["Rule"] != "Fallback treatments"]
        return [{"section": "General rule", "order": position, "item": row["Rule"], "setting": str(row["Setting"])}
                for position, row in general.iterrows()]