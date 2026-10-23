# SACCO loan collections

A collections system for a Ugandan SACCO (savings and credit co-operative), built as the final project for the Python OOP course. It scores each loan's probability of default (PD) and puts it in a risk tier, groups overdue loans into delinquency buckets by days past due (DPD) and months in arrears (MIA), ranks them for loan officers, runs a daily collections process that checks compliance rules before contacting anyone, logs every message, projects where today's balance will be over the next 12 months, and shows it all on a dashboard. The notebook `sacco.ipynb` walks through it step by step.

All data is synthetic. `SyntheticCoreBanking` simulates 1,200 members saving and borrowing from October 2023 to the business date, 22 October 2026, so balances, arrears, promises and collection activity agree with each other.

## Setup

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

Start Jupyter from this folder, because the notebook reads `data/`, `kb/` and `configs/` by relative path:

```bash
jupyter lab sacco.ipynb
```

Run the cells from the top. The notebook goes from the source data, through the data itself (customers, cases and their histories), to the collections policy, the risk model, portfolio profiling, the collections worklist, the daily collections run with its notification log, the debt projection and finally the dashboard. `AS_OF`, set in the first data cell, is the business date the data describes; every list and the model use it, so results don't change with the day you run the notebook. The lists are interactive (ipywidgets), so open the notebook in Jupyter or VS Code rather than a static viewer.

- The data files in `data/` are already generated. Set `REGENERATE = True` in the "Source data" cell to rebuild them; the fixed seed (2026) gives byte-for-byte the same files, and a test checks this.
- The "Risk model" section trains the PD model and saves it to `models/pd_model.joblib`. The file is a pickle of scikit-learn objects: load it only with the same library versions, and only load files you created.

## Running the tests

```bash
pytest
```

`pytest.ini` points pytest at `tests/` and puts this folder on the import path. The full suite takes about 20 seconds. It needs no Jupyter kernel: widgets are built but not drawn, and charts use the non-interactive Agg backend.

| Test file | What it covers |
|-----------|----------------|
| `test_csv_import.py` | Loading CSV files and folders, and every error for a bad file |
| `test_policy.py` | Bucket and risk-tier edges, profile priority, treatment lookup, each policy mistake the loader rejects, and missing settings raising `PolicyError` |
| `test_customers.py` | The shared customer-name helper and the `HistoryList` base class |
| `test_display.py` | The `Display` base class and its filters: filtering, sorting, paging, reset |
| `test_lists.py` | Every list in the notebook, including the worklist ranking, on the real data and on small edge-case rows |
| `test_risk_model.py` | Feature building without look-ahead, label correctness, training, evaluation, save and load |
| `test_collections_run.py` | Each compliance rule, the compliance checker, the dunning engine and the profiler |
| `test_notifications.py` | Each channel's message, sent, queued, referred and blocked entries, and the log of a real run |
| `test_roll_rate_model.py` | Roll rates on hand-checked snapshots, projection, expected write-offs, and the back-test |
| `test_dashboard.py` | Dashboard figures, filters, the projection, HTML rendering, and the CSV and PDF exports |
| `test_synthetic_core_banking.py` | The generator is reproducible and consistent, computes MIA from the monthly instalment, and matches `data/` |
| `test_notebook.py` | Every notebook code cell parses and every `src` import in it exists |

## Project layout

```
sacco.ipynb             the walkthrough notebook
configs/policy.json     the collections policy: buckets, risk tiers, profiles, treatment matrix, compliance limits
data/                   the SACCO's records (generated): customers, loans, payments, contacts, promises, treatments, monthly loan snapshots
kb/                     reference data: loan products, treatment types, contact and address types, Uganda locations
models/                 the trained PD model
export/                 dashboard exports, created when you export (ignored by git)
src/
  csv_import.py         CsvImport: reads CSV files into rows and records
  customers.py          customer names for every list, in one place
  exceptions.py         SaccoError and one subclass per kind of failure
  policy.py             Policy: loads and checks policy.json, answers bucket/tier/profile/treatment questions; its raw settings are private
  synthetic_core_banking.py   SyntheticCoreBanking: generates data/
  display/              the searchable lists shown in the notebook
  risk/                 FeatureBuilder, RiskModel, LogisticPdModel, RollRateModel
  profiling.py          Profiler: bucket, PD, tier and profile for every open loan on every business date
  dunning.py            DunningEngine: which treatment each overdue loan gets
  compliance/           the contact rules and the ComplianceChecker that applies them
  notifications.py      message templates per channel, and the NotificationLog
  dashboard/            DashboardData, the dashboard widget, CSV and PDF exporters
tests/                  pytest suite
```

## How the parts fit together

1. `CsvImport` loads `kb/` and `data/` into dictionaries of rows.
2. `Policy` loads `configs/policy.json` and refuses a policy with gaps, overlaps or unknown names, so a mistake stops the run instead of treating customers the wrong way.
3. The `display` lists show customers, cases, the policy and each history file.
4. `FeatureBuilder` turns the monthly loan snapshots into one row per open loan per month, using only what was known that day. `LogisticPdModel` learns from earlier months and is tested on later ones.
5. `Profiler` combines the policy, the PD and each customer's reachability to give every open loan a DPD bucket, an MIA bucket, a risk tier and a profile.
6. `Worklist` ranks the overdue loans by expected loss (PD times outstanding balance), filterable by officer, bucket, tier and profile.
7. `DunningEngine` proposes a treatment per overdue loan, falling back (for example SMS to email) when the customer can't be reached on the first channel.
8. `ComplianceChecker` runs each proposal through the contact rules (case status, pending promise, contact hours, channel limits, minimum gap, daily cap) and records what was blocked and why.
9. `NotificationLog` writes every decision as a message: sent, queued for an officer, referred, or blocked with the reason. `NotificationList` shows the log.
10. `RollRateModel` learns how balance moves between buckets each month (a Markov chain) and projects today's loans 3, 6 and 12 months ahead, including new write-offs.
11. `DashboardData` and `PortfolioDashboard` show the portfolio for any business date and export it as CSV or PDF.

Every step is in the notebook.

## Object-oriented design

Each extension point is an abstract base class, so a new kind of list, model, rule or export format is a new subclass and nothing else changes:

| Base class | Subclasses | What a subclass provides |
|------------|-----------|--------------------------|
| `Display` | `CustomerList`, `CaseList`, `HistoryList` (below), `BucketList`, `RiskTierList`, `ProfileRuleList`, `TreatmentList`, `ChannelLimitList`, `GeneralRuleList`, `Worklist`, `NotificationList` | The table (`build_frame`) and its filters (`build_filters`); `Display` does widgets, filtering, sorting and paging |
| `HistoryList` | `ContactHistory`, `PaymentHistory`, `PtpHistory`, `TreatmentHistory` | Its own rows; `HistoryList` adds the account number and customer name and rejects rows for cases that don't exist |
| `Filter` | `TextFilter`, `ChoiceFilter`, `RangeFilter` | Its widget, which rows it keeps, how it resets |
| `RiskModel` | `LogisticPdModel` | `fit` and `predict_pd`; `RiskModel` does evaluation, score bands, save and load |
| `ContactRule` | `CaseStatusRule`, `PromisePendingRule`, `ContactHoursRule`, `DailyChannelLimitRule`, `WeeklyChannelLimitRule`, `MinimumGapRule`, `DailyContactCapRule` | One `check` that allows or blocks an action with a reason |
| `MessageTemplate` | `SmsTemplate`, `EmailTemplate`, `CallTemplate`, `ReferralTemplate` | Who a channel's message goes to and what it says |
| `ReportExporter` | `CsvExporter`, `PdfExporter` | `export` for one format |

The errors the system raises on purpose are subclasses of `SaccoError` (`ImportFileError`, `DisplayError`, `PolicyError`, `ModelError`, `NotificationError`, `ExportError`), so the notebook can catch them together or one kind at a time. A policy with a missing or misshapen setting raises `PolicyError` naming it (for example "Bucket 2 is missing 'upper'"), never a bare `KeyError`; the notebook's "When the policy is wrong" cell shows this.

`Policy` keeps the parsed file private. Other classes read it through properties (`buckets`, `risk_tiers`, `profiles`, `default_profile`, `fallback_treatments`, `treatment_matrix`, `compliance`) and methods (`bucket_for`, `tier_for`, `profile_for`, `treatment_for`, `describe`).

## Known issues

- The PD model's time split lets training rows from just before 31 December 2025 see outcomes inside the test months, so the reported AUC (about 0.91) is optimistic. Leaving a six-month gap between the training and test months gives about 0.87.
- The roll-rate projection is close to what happened up to six months ahead, but pessimistic at twelve: in the back-test from September 2025 it projected 2.8% of the balance written off against 1.2% actual, and too little paid off. Use the 12-month figure as an upper estimate.
- Expected loss on the worklist and dashboard assumes nothing is recovered from a defaulted loan, so it ranks loans rather than estimating a provision.

## Data sources

The Uganda district data in `kb/locations.csv` comes from the Uganda administrative boundaries dataset on the Humanitarian Data Exchange: https://data.humdata.org/dataset/cod-ab-uga. All people, phone numbers, national ID numbers and loans are invented.
