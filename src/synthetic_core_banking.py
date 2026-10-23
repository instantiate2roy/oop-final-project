"""
Generates the SACCO's synthetic data: customers, savings, loans, payments, collections activity
and monthly loan snapshots, written as CSV files in data/.

It stands in for a core banking system. Every file is built from one simulated history, so the
files agree with each other: balances equal what was paid, DPD and MIA follow from the instalment
schedule and the payments, promises are kept or broken according to the payments that followed,
and treatments match the DPD the loan actually reached.

Each customer has a hidden risk level that drives how they repay. Part of that risk shows up in
things the SACCO can see (income, savings habits, guarantors), so a risk model has real
patterns to learn, and part of it stays hidden, as in real lending.
"""
import bisect
import calendar
import csv
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .csv_import import CsvImport

SURNAMES = ["Kirabo", "Tusiime", "Ahimbisibwe", "Mayanja", "Kyarisiima", "Rukundo", "Ainomugisha", "Akumu",
            "Mugisha", "Tumwine", "Kamugisha", "Nassali", "Candia", "Acen", "Kawooya", "Tumusiime", "Nangobi",
            "Kakooza", "Ainembabazi", "Nafuna", "Opolot", "Mugabi", "Kakai", "Nabirye", "Nakalema", "Apio",
            "Odongo", "Anguzu", "Mulindwa", "Ocen", "Okot", "Kato", "Ssempala", "Namukasa", "Byaruhanga",
            "Asiimwe", "Nalwanga", "Okello", "Atim", "Wasswa", "Nakato", "Babirye", "Musoke", "Lubega",
            "Ochieng", "Akello", "Mutebi", "Nabukenya", "Kiggundu", "Natukunda", "Twesigye", "Arinaitwe",
            "Owor", "Adong", "Ssekandi", "Namatovu", "Bwambale", "Masereka", "Kansiime", "Ndyamuhaki"]
MALE_NAMES = ["Martin", "Geoffrey", "Julius", "Henry", "Isaac", "Samuel", "Lawrence", "Collins", "Musa",
              "Kenneth", "Brian", "Gerald", "Godfrey", "Vincent", "George", "Twaha", "Michael", "David",
              "Charles", "Paul", "Ambrose", "Elijah", "Fred", "Francis", "Ivan", "Robert", "Timothy",
              "Arthur", "Hassan", "Abdul", "Swaibu", "Hamza", "Ronald", "Richard", "Joseph", "Daniel",
              "Edward", "Peter", "Raymond", "Stephen"]
FEMALE_NAMES = ["Pamela", "Hadijah", "Susan", "Betty", "Josephine", "Vivian", "Rebecca", "Rehema", "Edith",
                "Joan", "Mercy", "Catherine", "Sarah", "Diana", "Martha", "Gloria", "Shamim", "Mary",
                "Margaret", "Olivia", "Hope", "Brenda", "Prossy", "Faridah", "Patience", "Esther", "Grace",
                "Rose", "Scovia", "Ruth", "Proscovia", "Lillian", "Sumaya", "Fatuma", "Christine",
                "Immaculate", "Zainab", "Sylvia", "Dorothy", "Winnie"]
ROADS = ["Central", "Katale", "Kirinya", "Station", "Market", "Church", "Mosque", "Hospital", "School",
         "Kampala", "Jinja", "Masaka", "Gulu", "Mbarara", "Entebbe", "Hoima", "Lira", "Mbale"]
LANDMARKS = ["primary school", "health centre", "trading centre", "church", "mosque", "market", "borehole"]

EMPLOYMENT = {          # share of members, median monthly income (UGX), name
    "SALARIED": (0.30, 1_300_000, "salaried employee"),
    "SELF_EMPLOYED": (0.35, 950_000, "self-employed / business owner"),
    "FARMER": (0.22, 520_000, "farmer"),
    "CASUAL": (0.13, 380_000, "casual worker"),
}
LOAN_PRODUCTS = {       # monthly rate %, income multiple range, risk effect, which members take it
    "business":       (2.5, (2.0, 6.0), 0.15, {"SELF_EMPLOYED": 5, "SALARIED": 1, "FARMER": 1, "CASUAL": 1}),
    "agriculture":    (2.0, (1.0, 3.0), 0.30, {"FARMER": 6, "SELF_EMPLOYED": 1}),
    "school_fees":    (2.0, (0.8, 1.6), 0.00, {"SALARIED": 3, "SELF_EMPLOYED": 3, "FARMER": 3, "CASUAL": 2}),
    "salary":         (1.5, (3.0, 8.0), -0.90, {"SALARIED": 5}),
    "emergency":      (3.0, (0.4, 1.0), 0.20, {"SALARIED": 2, "SELF_EMPLOYED": 2, "FARMER": 2, "CASUAL": 4}),
    "asset":          (2.5, (3.0, 6.0), 0.10, {"SALARIED": 1, "SELF_EMPLOYED": 2, "CASUAL": 1}),
    "housing":        (1.5, (8.0, 18.0), -0.30, {"SALARIED": 1, "SELF_EMPLOYED": 1}),
    "savings_backed": (1.0, (0.5, 1.5), -1.20, {"SALARIED": 1, "SELF_EMPLOYED": 1, "FARMER": 1, "CASUAL": 2}),
    "group":          (2.0, (2.0, 4.0), -0.20, {"FARMER": 2, "CASUAL": 3, "SELF_EMPLOYED": 1}),
}
PAYMENT_CHANNELS = [("MTN_MOMO", 0.42), ("AIRTEL_MONEY", 0.24), ("CASH", 0.20), ("BANK_DEPOSIT", 0.12),
                    ("SAVINGS_OFFSET", 0.02)]


@dataclass
class GeneratorSettings:
    seed: int = 2026
    as_of: date = date(2026, 10, 22)          # the business date the snapshot is taken on
    loans_from: date = date(2024, 10, 1)      # first possible disbursement
    savings_from: date = date(2023, 10, 1)    # savings history starts here
    customers: int = 1200


@dataclass
class Member:
    uuid: str
    sex: str
    surname: str
    given_name: str
    dob: date
    employment: str
    income: int
    member_since: date
    risk: float                 # hidden: higher means more likely to fall behind
    savings_regularity: float   # chance of saving in a given month
    reachability: float         # chance of answering a call
    home_district: str
    mobile_uuid: str | None = None
    email_uuid: str | None = None
    third_party_uuid: str | None = None
    savings: list = field(default_factory=list)   # (date, amount, balance_after)


@dataclass
class Loan:
    uuid: str
    member: Member
    product: str
    disbursed: date
    amount: int
    rate: float
    term: int
    frequency: str
    total: int
    due_dates: list
    due_amounts: list
    payments: list = field(default_factory=list)  # (date, amount, channel, reference, status)
    guarantor_cover: float = 0.0
    written_off: date | None = None
    legal: date | None = None
    account_no: str = ""

    # ---- ledger: everything follows from the schedule and the posted payments ----
    def build_ledger(self) -> None:
        self._due_ord = [d.toordinal() for d in self.due_dates]
        self._due_cum = list(np.cumsum(self.due_amounts))
        posted = sorted((p[0], p[1]) for p in self.payments if p[4] == "POSTED")
        self._pay_ord = [p[0].toordinal() for p in posted]
        self._pay_cum = list(np.cumsum([p[1] for p in posted])) if posted else []

    def paid_by(self, day: date) -> int:
        i = bisect.bisect_right(self._pay_ord, day.toordinal())
        return int(self._pay_cum[i - 1]) if i else 0

    def due_by(self, day: date) -> int:
        """Instalments due before `day`; one falling due on `day` itself isn't in arrears yet."""
        i = bisect.bisect_left(self._due_ord, day.toordinal())
        return int(self._due_cum[i - 1]) if i else 0

    def position(self, day: date) -> tuple[int, int, int]:
        """(dpd, arrears, outstanding) at the end of `day`."""
        paid = self.paid_by(day)
        arrears = max(0, self.due_by(day) - paid)
        outstanding = max(0, self.total - paid)
        dpd = 0
        if arrears:
            first_unpaid = bisect.bisect_right(self._due_cum, paid)
            dpd = day.toordinal() - self._due_ord[first_unpaid]
        return dpd, arrears, outstanding

    @property
    def instalment(self) -> int:
        return self.due_amounts[0]

    @property
    def monthly_instalment(self) -> float:
        """What the loan costs per month, whatever the repayment frequency; MIA divides arrears by this."""
        return self.total / self.term

    def mia(self, arrears: int) -> float:
        """Months in arrears: a weekly loan 4 weeks behind is about 1 month behind, not 4."""
        return round(arrears / self.monthly_instalment, 2)

    def settled_on(self) -> date | None:
        if not self._pay_cum or self._pay_cum[-1] < self.total:
            return None
        i = bisect.bisect_left(self._pay_cum, self.total)
        return date.fromordinal(self._pay_ord[i])


class SyntheticCoreBanking:
    """Simulates the SACCO's members and loans and writes the data files."""

    def __init__(self, settings: GeneratorSettings | None = None, kb_folder: str | Path = "kb"):
        self.settings = settings or GeneratorSettings()
        self.rng = np.random.default_rng(self.settings.seed)
        kb = CsvImport().load_dir_data(kb_folder)
        self.loan_types = {row["code"]: row for row in kb["loan_types"]["rows"]}
        self.locations = kb["locations"]["rows"]
        self.locations_by_district: dict[str, list[dict]] = {}
        for location in self.locations:
            self.locations_by_district.setdefault(location["district"], []).append(location)
        self.districts = sorted(self.locations_by_district)
        self.rows: dict[str, list[dict]] = {name: [] for name in [
            "customer", "address", "contact", "savings_history", "case", "payment_history", "security",
            "contact_history", "ptp", "treatment_history", "case_snapshot"]}
        self.members: list[Member] = []
        self._pending_ptp: set[str] = set()
        self.loans: list[Loan] = []

    # ---- small helpers --------------------------------------------------------
    def _uuid(self) -> str:
        return str(uuid.UUID(bytes=self.rng.bytes(16), version=4))

    def _choice(self, options, weights=None):
        weights = np.array(weights, dtype=float) if weights is not None else None
        return options[self.rng.choice(len(options), p=None if weights is None else weights / weights.sum())]

    @staticmethod
    def _add_months(day: date, months: int) -> date:
        month = day.month - 1 + months
        year, month = day.year + month // 12, month % 12 + 1
        return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _random_date(self, start: date, end: date) -> date:
        return start + timedelta(days=int(self.rng.integers(0, max(1, (end - start).days + 1))))

    def _working_datetime(self, day: date) -> datetime:
        """A time between 08:00 and 17:59 on a working day (Sunday moves to Monday)."""
        if day.weekday() == 6:
            day += timedelta(days=1)
        return datetime(day.year, day.month, day.day, int(self.rng.integers(8, 18)), int(self.rng.integers(0, 60)))

    @staticmethod
    def _ugx(amount: int) -> str:
        return f"UGX {amount:,}"

    # ---- members --------------------------------------------------------------
    def _make_members(self) -> None:
        s = self.settings
        kinds = list(EMPLOYMENT)
        for _ in range(s.customers):
            sex = "M" if self.rng.random() < 0.5 else "F"
            employment = self._choice(kinds, [EMPLOYMENT[k][0] for k in kinds])
            risk = float(self.rng.normal())
            income = int(round(EMPLOYMENT[employment][1] * math.exp(self.rng.normal(0, 0.45)) / 10_000) * 10_000)
            dob = self._random_date(date(1962, 1, 1), date(2004, 12, 31))
            member = Member(
                uuid=self._uuid(), sex=sex, surname=self._choice(SURNAMES),
                given_name=self._choice(MALE_NAMES if sex == "M" else FEMALE_NAMES), dob=dob,
                employment=employment, income=max(income, 150_000),
                member_since=self._random_date(date(2014, 1, 1), date(2024, 9, 30)), risk=risk,
                savings_regularity=self._sigmoid(1.1 - 0.9 * risk + self.rng.normal(0, 0.7)),
                reachability=self._sigmoid(0.9 - 0.7 * risk + self.rng.normal(0, 0.6)),
                home_district=self._choice(self.districts),
            )
            self.members.append(member)
            self.rows["customer"].append({
                "uuid": member.uuid, "surname": member.surname, "given_name": member.given_name, "sex": sex,
                "nationality": "Ugandan", "dob": dob.isoformat(), "nin": self._fake_nin(sex, dob),
                "employment_type_id": employment, "monthly_income": member.income,
                "member_since": member.member_since.isoformat(),
            })
            self._make_addresses(member)
            self._make_contacts(member)
            self._make_savings(member)

    def _fake_nin(self, sex: str, dob: date) -> str:
        # clearly fake: underscores never appear in a real NIN
        letters = "".join(chr(65 + int(self.rng.integers(0, 26))) for _ in range(4))
        return f"C{sex}_{dob.year % 100:02d}{int(self.rng.integers(0, 10**6)):06d}{letters[:3]}_{letters[3]}"

    def _address_line(self, location: dict) -> str:
        style = self.rng.random()
        if style < 0.5:
            return f"Plot {int(self.rng.integers(1, 400))}, {self._choice(ROADS)} Road"
        place = location["village"].split(" Village")[0]
        return f"{self._choice(['Near', 'Behind', 'Opposite'])} {place} {self._choice(LANDMARKS)}"

    def _make_addresses(self, member: Member) -> None:
        kinds = [("HOME", member.home_district)]
        if member.employment == "SALARIED":
            kinds.append(("WORK", self._choice(["Kampala", "Wakiso", member.home_district], [3, 2, 2])))
        if member.employment in ("SELF_EMPLOYED", "FARMER") and self.rng.random() < 0.6:
            kinds.append(("BUSINESS", member.home_district))
        if self.rng.random() < 0.25:
            kinds.append(("ANCESTRAL_HOME", self._choice(self.districts)))
        for kind, district in kinds:
            location = self._choice(self.locations_by_district[district])
            status = self._choice(["VERIFIED", "UNVERIFIED", "MOVED", "NOT_FOUND"], [40, 48, 6, 6])
            self.rows["address"].append({
                "uuid": self._uuid(), "customer_uuid": member.uuid, "address_type_id": kind,
                "location_uuid": location["uuid"], "address_line": self._address_line(location), "status": status})

    def _phone(self) -> str:
        return f"+2567{self._choice(['0', '4', '5', '6', '7', '8'])}{int(self.rng.integers(0, 10**7)):07d}"

    def _add_contact(self, member: Member, kind: str, value: str, valid: bool) -> str:
        contact_uuid = self._uuid()
        self.rows["contact"].append({"uuid": contact_uuid, "customer_uuid": member.uuid, "contact_type_id": kind,
                                     "value": value, "valid": "Y" if valid else "N"})
        return contact_uuid

    def _make_contacts(self, member: Member) -> None:
        main_valid = self.rng.random() < 0.95
        mobile = self._phone()
        main = self._add_contact(member, "MOBILE_PHONE", mobile, main_valid)
        member.mobile_uuid = main if main_valid else None
        if self.rng.random() < 0.45:
            second_valid = self.rng.random() < 0.7
            second = self._add_contact(member, "MOBILE_PHONE", self._phone(), second_valid)
            member.mobile_uuid = member.mobile_uuid or (second if second_valid else None)
        if self.rng.random() < 0.4:
            self._add_contact(member, "WHATSAPP", mobile, main_valid)
        email_chance = {"SALARIED": 0.7, "SELF_EMPLOYED": 0.3, "FARMER": 0.1, "CASUAL": 0.1}[member.employment]
        if self.rng.random() < email_chance:
            valid = self.rng.random() < 0.92
            handle = f"{member.given_name}.{member.surname}{int(self.rng.integers(1, 99))}".lower()
            email = self._add_contact(member, "EMAIL", f"{handle}@{self._choice(['gmail.com', 'yahoo.com'])}", valid)
            member.email_uuid = email if valid else None
        if member.employment == "SALARIED" and self.rng.random() < 0.3:
            self._add_contact(member, "WORK_PHONE", self._phone(), True)
        for kind, chance in [("NEXT_OF_KIN_PHONE", 0.6), ("SPOUSE_PHONE", 0.4), ("REFEREE_PHONE", 0.15)]:
            if self.rng.random() < chance:
                contact_uuid = self._add_contact(member, kind, self._phone(), self.rng.random() < 0.9)
                member.third_party_uuid = member.third_party_uuid or contact_uuid

    def _make_savings(self, member: Member) -> None:
        start = max(member.member_since, self.settings.savings_from)
        month, balance = date(start.year, start.month, 1), int(member.income * self.rng.uniform(0.2, 1.5))
        balance = int(round(balance / 1000) * 1000)
        while month <= self.settings.as_of:
            if self.rng.random() < member.savings_regularity:
                day = self._random_date(month, min(self._add_months(month, 1) - timedelta(days=1), self.settings.as_of))
                amount = int(round(member.income * self.rng.uniform(0.03, 0.12) / 1000) * 1000)
                balance += amount
                member.savings.append((day, amount, balance))
                self.rows["savings_history"].append({
                    "uuid": self._uuid(), "customer_uuid": member.uuid, "transaction_date": day.isoformat(),
                    "transaction_type_id": "DEPOSIT", "amount": amount, "balance_after": balance})
            if balance > 0 and self.rng.random() < 0.05:
                day = self._random_date(month, min(self._add_months(month, 1) - timedelta(days=1), self.settings.as_of))
                amount = int(round(balance * self.rng.uniform(0.1, 0.5) / 1000) * 1000)
                if amount:
                    balance -= amount
                    member.savings.append((day, -amount, balance))
                    self.rows["savings_history"].append({
                        "uuid": self._uuid(), "customer_uuid": member.uuid, "transaction_date": day.isoformat(),
                        "transaction_type_id": "WITHDRAWAL", "amount": amount, "balance_after": balance})
            month = self._add_months(month, 1)

    def _savings_on(self, member: Member, day: date) -> int:
        balance = 0
        for when, _, after in member.savings:
            if when <= day:
                balance = after
        return balance

    # ---- loans ----------------------------------------------------------------
    def _make_loans(self) -> None:
        s = self.settings
        borrowers = [m for m in self.members if self.rng.random() < 0.85]   # the rest only save or guarantee
        for member in borrowers:
            day = self._random_date(s.loans_from, s.loans_from + timedelta(days=420))
            while day <= s.as_of - timedelta(days=10):
                loan = self._simulate_loan(member, day)
                self.loans.append(loan)
                if loan.written_off or loan.legal:
                    break                                           # no new loans after a default
                closed = loan.settled_on()
                if closed is None or self.rng.random() < 0.35:
                    break
                day = closed + timedelta(days=int(self.rng.integers(7, 120)))

    def _pick_product(self, member: Member) -> str:
        products = [p for p, spec in LOAN_PRODUCTS.items() if member.employment in spec[3]]
        return self._choice(products, [LOAN_PRODUCTS[p][3][member.employment] for p in products])

    def _schedule(self, start: date, total: int, term: int, frequency: str) -> tuple[list, list]:
        if frequency == "monthly":
            dates = [self._add_months(start, k) for k in range(1, term + 1)]
        elif frequency == "weekly":
            dates = [start + timedelta(weeks=k) for k in range(1, term * 4 + 1)]
        else:  # lump_sum
            dates = [self._add_months(start, term)]
        each = int(round(total / len(dates) / 100) * 100)
        amounts = [each] * (len(dates) - 1) + [total - each * (len(dates) - 1)]
        return dates, amounts

    def _simulate_loan(self, member: Member, disbursed: date) -> Loan:
        product = self._pick_product(member)
        rate, (low, high), product_risk, _ = LOAN_PRODUCTS[product]
        spec = self.loan_types[product]
        term = int(self.rng.integers(int(spec["min_term_months"]), int(spec["max_term_months"]) + 1))
        if product == "agriculture":
            frequency = "lump_sum"
        elif product in ("group", "asset") and self.rng.random() < 0.4:
            frequency = "weekly"
        else:
            frequency = "monthly"
        amount = member.income * self.rng.uniform(low, high)
        if product == "savings_backed":
            amount = max(200_000, min(amount, self._savings_on(member, disbursed) * 1.5))
        amount = int(max(200_000, round(amount / 50_000) * 50_000))
        total = int(round(amount * (1 + rate / 100 * term) / 100) * 100)
        due_dates, due_amounts = self._schedule(disbursed, total, term, frequency)
        loan = Loan(uuid=self._uuid(), member=member, product=product, disbursed=disbursed, amount=amount,
                    rate=rate, term=term, frequency=frequency, total=total, due_dates=due_dates,
                    due_amounts=due_amounts)
        self._make_security(loan)

        monthly_burden = total / term
        dti = monthly_burden / member.income
        savings_gap = 0.6 - member.savings_regularity
        logit = (-1.0 + 1.0 * member.risk + 1.6 * (dti - 0.4) + 1.4 * savings_gap
                 - 0.45 * min(loan.guarantor_cover, 2.0) + product_risk
                 + {"CASUAL": 0.4, "FARMER": 0.15, "SELF_EMPLOYED": 0.0, "SALARIED": -0.3}[member.employment])
        trouble_chance = self._sigmoid(logit)
        self._simulate_repayment(loan, trouble_chance)
        return loan

    def _channel_and_reference(self, loan: Loan) -> tuple[str, str]:
        if loan.product == "salary":
            return "PAYROLL", f"PAY{int(self.rng.integers(10**5, 10**6))}"
        channel = self._choice([c for c, _ in PAYMENT_CHANNELS], [w for _, w in PAYMENT_CHANNELS])
        reference = {"MTN_MOMO": f"AM{int(self.rng.integers(10**8, 10**9))}",
                     "AIRTEL_MONEY": f"{int(self.rng.integers(10**9, 10**10))}",
                     "CASH": f"RCT{int(self.rng.integers(10**5, 10**6))}",
                     "BANK_DEPOSIT": f"BD{int(self.rng.integers(10**6, 10**7))}",
                     "SAVINGS_OFFSET": f"JV{int(self.rng.integers(10**4, 10**5))}"}[channel]
        return channel, reference

    def _pay(self, loan: Loan, day: date, amount: int) -> None:
        if amount <= 0 or day > self.settings.as_of:
            return
        channel, reference = self._channel_and_reference(loan)
        if self.rng.random() < 0.015:   # a failed attempt, retried a couple of days later
            loan.payments.append((day, amount, channel, reference, "FAILED"))
            day += timedelta(days=int(self.rng.integers(1, 4)))
            if day > self.settings.as_of:
                return
            channel, reference = self._channel_and_reference(loan)
        loan.payments.append((day, amount, channel, reference, "POSTED"))

    def _simulate_repayment(self, loan: Loan, trouble_chance: float) -> None:
        """Walks through the due dates (then monthly after maturity) deciding what the borrower pays."""
        as_of, member = self.settings.as_of, loan.member
        periods = len(loan.due_dates)
        per_period = 1 - (1 - trouble_chance) ** (1 / max(1, periods)) if trouble_chance < 1 else 1
        troubled, paid, owed = False, 0, 0
        late_scale = 2 + 4 * max(0.0, member.risk)
        checkpoints = list(zip(loan.due_dates, loan.due_amounts))
        k = 0
        while k < len(checkpoints) or (paid < loan.total and checkpoints[-1][0] <= as_of and k < len(checkpoints) + 18):
            if k < len(checkpoints):
                due, amount = checkpoints[k]
                owed += amount
            else:   # after maturity: one more month to catch up
                due = self._add_months(checkpoints[-1][0], k - len(checkpoints) + 1)
            if due > as_of:
                break
            if not troubled and self.rng.random() < per_period:
                troubled = True
            next_due = checkpoints[k + 1][0] if k + 1 < len(checkpoints) else self._add_months(due, 1)
            if not troubled:
                delay = 0 if self.rng.random() < 0.6 else int(self.rng.geometric(1 / late_scale))
                if self.rng.random() < 0.06:            # pays part now, the rest at the next due date
                    part = int(round((owed - paid) * self.rng.uniform(0.5, 0.9) / 100) * 100)
                    self._pay(loan, due + timedelta(days=delay), part)
                    paid += part
                else:
                    self._pay(loan, due + timedelta(days=delay), owed - paid)
                    paid = owed
            else:
                roll = self.rng.random()
                cure = 0.16 - 0.05 * member.risk
                if roll < cure:                         # catches up in full and recovers
                    day = due + timedelta(days=int(self.rng.integers(3, 25)))
                    self._pay(loan, min(day, next_due), owed - paid)
                    paid = owed
                    troubled = self.rng.random() < 0.25
                elif roll < cure + 0.35 and owed > paid:  # pays something
                    part = int(round(loan.instalment * self.rng.uniform(0.2, 0.7) / 100) * 100)
                    part = min(part, owed - paid)
                    self._pay(loan, due + timedelta(days=int(self.rng.integers(1, 25))), part)
                    paid += part
            k += 1
        loan.payments.sort(key=lambda p: p[0])
        loan.build_ledger()
        self._apply_default_outcomes(loan)

    def _apply_default_outcomes(self, loan: Loan) -> None:
        """Refers big defaults to legal at 120 DPD and writes off at 180 DPD; later payments stop."""
        day = loan.due_dates[0]
        while day <= self.settings.as_of:
            dpd, _, outstanding = loan.position(day)
            if outstanding == 0:
                return
            if dpd >= 120 and loan.amount >= 3_000_000 and loan.legal is None:
                loan.legal = day
            if dpd >= 180 and loan.legal is None:
                loan.written_off = day
                break
            day += timedelta(days=1)
        stop = loan.written_off or loan.legal
        if stop:
            loan.payments = [p for p in loan.payments if p[0] <= stop]
            loan.build_ledger()

    def _make_security(self, loan: Loan) -> None:
        guarantors = 0 if loan.product == "savings_backed" else int(self.rng.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.35, 0.1]))
        others = [m for m in self.members if m.uuid != loan.member.uuid]
        cover = 0
        for _ in range(guarantors):
            guarantor = others[int(self.rng.integers(0, len(others)))]
            value = int(round(loan.amount * self.rng.uniform(0.2, 0.6) / 10_000) * 10_000)
            cover += value
            self.rows["security"].append({"uuid": self._uuid(), "case_uuid": loan.uuid, "security_type_id": "GUARANTOR",
                                          "description": f"Guaranteed by {guarantor.given_name} {guarantor.surname}",
                                          "value": value, "guarantor_customer_uuid": guarantor.uuid, "status": ""})
        savings = self._savings_on(loan.member, loan.disbursed)
        if savings and (loan.product == "savings_backed" or self.rng.random() < 0.6):
            value = int(min(savings, loan.amount))
            cover += value
            self.rows["security"].append({"uuid": self._uuid(), "case_uuid": loan.uuid, "security_type_id": "SAVINGS",
                                          "description": f"Savings account SV{int(self.rng.integers(10**6, 10**7))}",
                                          "value": value, "guarantor_customer_uuid": "", "status": ""})
        extra = {"salary": ("CHECK_OFF", "Employer check-off deduction"), "housing": ("LAND_TITLE", "Land title"),
                 "asset": ("ASSET", "Financed asset"), "group": ("GROUP_GUARANTEE", "Group guarantee")}.get(loan.product)
        if extra:
            value = loan.amount if extra[0] != "GROUP_GUARANTEE" else int(loan.amount * 0.5)
            cover += value
            self.rows["security"].append({"uuid": self._uuid(), "case_uuid": loan.uuid, "security_type_id": extra[0],
                                          "description": extra[1], "value": value,
                                          "guarantor_customer_uuid": "", "status": ""})
        loan.guarantor_cover = cover / loan.amount

    # ---- collections activity -------------------------------------------------
    def _contact(self, loan: Loan, day: date, channel: str, outcome: str, notes: str, contact_uuid: str | None) -> str:
        history_uuid = self._uuid()
        self.rows["contact_history"].append({
            "uuid": history_uuid, "case_uuid": loan.uuid, "contact_uuid": contact_uuid or "",
            "contact_datetime": self._working_datetime(day).strftime("%Y-%m-%d %H:%M"),
            "channel": channel, "outcome": outcome, "notes": notes})
        return history_uuid

    def _message(self, loan: Loan, day: date, dpd: int, arrears: int) -> None:
        member = loan.member
        if member.mobile_uuid:
            outcome = self._choice(["SMS_DELIVERED", "SMS_SENT", "SMS_FAILED"], [82, 12, 6])
            self._contact(loan, day, "SMS", outcome, f"Your loan is {dpd} days overdue. Please pay {self._ugx(arrears)} today",
                          member.mobile_uuid)
        elif member.email_uuid:
            outcome = self._choice(["EMAIL_SENT", "EMAIL_BOUNCED"], [95, 5])
            self._contact(loan, day, "EMAIL", outcome, f"Arrears notice: {dpd} days overdue, {self._ugx(arrears)}",
                          member.email_uuid)

    def _call(self, loan: Loan, day: date, dpd: int, arrears: int) -> None:
        member = loan.member
        if not member.mobile_uuid:
            return
        upcoming = [p for p in loan.payments if p[4] == "POSTED" and day < p[0] <= day + timedelta(days=10)]
        answered = self.rng.random() < member.reachability
        if answered and upcoming and self.rng.random() < 0.65:
            payment = upcoming[0]
            promised = min(arrears, int(round(payment[1] * self.rng.uniform(0.9, 1.2) / 1000) * 1000)) or payment[1]
            history = self._contact(loan, day, "CALL", "PROMISED_TO_PAY",
                                    f"Promised {self._ugx(promised)} by {payment[0].isoformat()}", member.mobile_uuid)
            self._promise(loan, day, payment[0], promised, history)
        elif answered:
            outcome = self._choice(["RIGHT_PARTY_CONTACT", "PROMISED_TO_PAY", "REFUSED_TO_PAY", "CALL_BACK"], [45, 15, 15, 25])
            if outcome == "PROMISED_TO_PAY":
                promised_day = day + timedelta(days=int(self.rng.integers(3, 11)))
                promised = int(round(arrears * self.rng.uniform(0.4, 1.0) / 1000) * 1000) or arrears
                history = self._contact(loan, day, "CALL", outcome,
                                        f"Promised {self._ugx(promised)} by {promised_day.isoformat()}", member.mobile_uuid)
                self._promise(loan, day, promised_day, promised, history)
            else:
                notes = {"RIGHT_PARTY_CONTACT": f"Reminded of arrears of {self._ugx(arrears)}",
                         "REFUSED_TO_PAY": self._choice(["Disputes the balance", "Lost job, cannot pay now",
                                                         "Says the business has collapsed", "Says the guarantor should pay",
                                                         "Will only pay after harvest"]),
                         "CALL_BACK": "Asked to call back later"}[outcome]
                self._contact(loan, day, "CALL", outcome, notes, member.mobile_uuid)
        else:
            outcome = self._choice(["NO_ANSWER", "SWITCHED_OFF", "BUSY", "THIRD_PARTY_CONTACT", "WRONG_NUMBER"], [45, 25, 15, 10, 5])
            if outcome == "THIRD_PARTY_CONTACT" and member.third_party_uuid:
                self._contact(loan, day, "CALL", outcome, "Spoke to the customer's next of kin, who will pass on the message",
                              member.third_party_uuid)
            else:
                self._contact(loan, day, "CALL", outcome if outcome != "THIRD_PARTY_CONTACT" else "NO_ANSWER", "",
                              member.mobile_uuid)

    def _promise(self, loan: Loan, made: date, promised_day: date, amount: int, history_uuid: str) -> None:
        ptp_uuid = self._uuid()
        paid = sum(p[1] for p in loan.payments if p[4] == "POSTED" and made < p[0] <= promised_day)
        if promised_day > self.settings.as_of:
            status = "PENDING"
        elif paid >= amount:
            status = "KEPT"
        elif paid > 0:
            status = "PARTIALLY_KEPT"
        else:
            status = "BROKEN"
        self.rows["ptp"].append({"uuid": ptp_uuid, "case_uuid": loan.uuid, "contact_history_uuid": history_uuid,
                                 "created_date": made.isoformat(), "promised_date": promised_day.isoformat(),
                                 "promised_amount": amount, "paid_amount": paid, "status": status})
        for i, p in enumerate(loan.payments):    # link the first payment made against the promise
            if p[4] == "POSTED" and made < p[0] <= promised_day and len(p) == 5:
                loan.payments[i] = (*p, ptp_uuid)
                break

    def _treatment(self, loan: Loan, kind: str, start: date, dpd: int, arrears: int) -> dict:
        assigned = {"DCA": f"DCA-0{int(self.rng.integers(1, 4))}", "LEGAL": "Legal department"}.get(kind, "In-house")
        row = {"uuid": self._uuid(), "case_uuid": loan.uuid, "treatment_type_id": kind, "start_date": start.isoformat(),
               "end_date": "", "dpd_at_start": dpd, "arrears_at_start": arrears, "assigned_to": assigned, "status": "ACTIVE"}
        self.rows["treatment_history"].append(row)
        return row

    def _make_collections(self, loan: Loan) -> None:
        as_of, member = self.settings.as_of, loan.member
        for due, amount in zip(loan.due_dates, loan.due_amounts):     # reminders before some due dates
            remind = due - timedelta(days=3)
            if loan.frequency != "weekly" and member.mobile_uuid and remind <= as_of and self.rng.random() < 0.3:
                self._contact(loan, remind, "SMS", "SMS_DELIVERED",
                              f"Reminder: instalment of {self._ugx(amount)} is due on {due.isoformat()}", member.mobile_uuid)

        end = min(as_of, loan.written_off or as_of)
        day, active, stage = loan.due_dates[0], None, None
        stages = [(1, "SMS"), (31, "CALL"), (61, "DCA"), (120, "LEGAL")]
        while day <= end:
            dpd, arrears, outstanding = loan.position(day)
            if dpd == 0:
                if active:
                    active.update(end_date=day.isoformat(), status="COMPLETED")
                    active, stage = None, None
                if outstanding == 0:
                    break
                day += timedelta(days=1)
                continue
            wanted = [kind for threshold, kind in stages if dpd >= threshold][-1]
            if wanted == "LEGAL" and loan.legal is None:
                wanted = "DCA"
            if wanted == "SMS" and not member.mobile_uuid:
                wanted = "EMAIL" if member.email_uuid else "CALL"
            if wanted != stage:
                if active:
                    active.update(end_date=day.isoformat(), status="ESCALATED")
                active, stage = self._treatment(loan, wanted, day, dpd, arrears), wanted
            if stage in ("SMS", "EMAIL") and dpd % 7 == 1:
                self._message(loan, day, dpd, arrears)
            elif stage in ("CALL", "DCA") and dpd % int(self.rng.integers(4, 8)) == 0:
                self._call(loan, day, dpd, arrears)
            day += timedelta(days=1)
        if active and loan.written_off and loan.written_off <= as_of:
            active.update(end_date=loan.written_off.isoformat(), status="STOPPED")

    # ---- writing the case and snapshot rows -----------------------------------
    def _case_status(self, loan: Loan, dpd: int, outstanding: int, day: date) -> tuple[str, str]:
        if outstanding == 0:
            return "CLOSED", "SETTLED"
        if loan.written_off and loan.written_off <= day:
            return "CLOSED", "WRITTEN_OFF"
        if loan.legal and loan.legal <= day:
            return "OPEN", "LEGAL"
        if dpd == 0:
            return "OPEN", "CURRENT"
        if day == self.settings.as_of and loan.uuid in self._pending_ptp:
            return "OPEN", "PROMISE_ACTIVE"
        return "OPEN", "NEW" if dpd <= 3 else "IN_PROGRESS"

    def _write_case(self, loan: Loan) -> None:
        as_of = self.settings.as_of
        dpd, arrears, outstanding = loan.position(as_of)
        account_status, status = self._case_status(loan, dpd, outstanding, as_of)
        posted = [p for p in loan.payments if p[4] == "POSTED"]
        closed = loan.settled_on() if status == "SETTLED" else loan.written_off if status == "WRITTEN_OFF" else None
        upcoming = [d for d in loan.due_dates if d > as_of]
        self.rows["case"].append({
            "uuid": loan.uuid, "customer_uuid": loan.member.uuid, "loan_type_uuid": self.loan_types[loan.product]["uuid"],
            "account_no": loan.account_no, "disbursement_date": loan.disbursed.isoformat(), "loan_amount": loan.amount,
            "monthly_interest_rate": loan.rate, "term_months": loan.term, "repayment_frequency": loan.frequency,
            "instalment_amount": loan.instalment, "total_payable": loan.total,
            "maturity_date": loan.due_dates[-1].isoformat(), "amount_paid": loan.paid_by(as_of),
            "outstanding_balance": outstanding, "arrears_amount": arrears, "dpd": dpd,
            "mia": loan.mia(arrears),
            "last_payment_date": posted[-1][0].isoformat() if posted else "",
            "last_payment_amount": posted[-1][1] if posted else "",
            "next_due_date": upcoming[0].isoformat() if upcoming and account_status == "OPEN" else "",
            "account_status": account_status, "status": status, "closed_date": closed.isoformat() if closed else ""})
        for p in loan.payments:
            self.rows["payment_history"].append({
                "uuid": self._uuid(), "case_uuid": loan.uuid, "payment_date": p[0].isoformat(), "amount": p[1],
                "channel": p[2], "reference": p[3], "status": p[4], "ptp_uuid": p[5] if len(p) > 5 else ""})
        for row in self.rows["security"]:
            if row["case_uuid"] == loan.uuid:
                row["status"] = self._security_status(row["security_type_id"], status)

    @staticmethod
    def _security_status(kind: str, case_status: str) -> str:
        if case_status == "SETTLED":
            return "RELEASED"
        if case_status in ("WRITTEN_OFF", "LEGAL"):
            return {"ASSET": "SOLD", "LAND_TITLE": "PLEDGED"}.get(kind, "CALLED")
        return "PLEDGED"

    def _write_snapshots(self, loan: Loan) -> None:
        month_end = date(loan.disbursed.year, loan.disbursed.month, calendar.monthrange(loan.disbursed.year, loan.disbursed.month)[1])
        while month_end <= self.settings.as_of:
            dpd, arrears, outstanding = loan.position(month_end)
            account_status, status = self._case_status(loan, dpd, outstanding, month_end)
            self.rows["case_snapshot"].append({
                "case_uuid": loan.uuid, "snapshot_date": month_end.isoformat(), "dpd": dpd,
                "mia": loan.mia(arrears), "arrears_amount": arrears,
                "outstanding_balance": outstanding, "account_status": account_status, "status": status})
            if account_status == "CLOSED":
                break
            nxt = self._add_months(month_end.replace(day=1), 1)
            month_end = date(nxt.year, nxt.month, calendar.monthrange(nxt.year, nxt.month)[1])

    # ---- run ------------------------------------------------------------------
    def generate(self) -> dict[str, list[dict]]:
        self._make_members()
        self._make_loans()
        self.loans.sort(key=lambda loan: loan.disbursed)
        for sequence, loan in enumerate(self.loans, start=1):
            loan.account_no = f"LN{loan.disbursed:%y%m}-{sequence:05d}"
            self._make_collections(loan)
        self._pending_ptp = {r["case_uuid"] for r in self.rows["ptp"] if r["status"] == "PENDING"}
        for loan in self.loans:
            self._write_case(loan)
            self._write_snapshots(loan)
        self.rows["contact_history"].sort(key=lambda r: r["contact_datetime"])
        return self.rows

    def write(self, folder: str | Path = "data") -> dict[str, int]:
        """Generate everything and write one CSV per table. Returns the row count per file."""
        if not any(self.rows.values()):
            self.generate()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        counts = {}
        for name, rows in self.rows.items():
            with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            counts[name] = len(rows)
        return counts
