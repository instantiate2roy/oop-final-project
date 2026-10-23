import pandas as pd
import pytest

from src.dashboard import CsvExporter, DashboardData, PdfExporter, PortfolioDashboard
from src.dashboard.report_exporter import exporter_for
from src.exceptions import ExportError

from conftest import AS_OF


@pytest.fixture(scope="module")
def dashboard(portfolio, data, roll_rates):
    return DashboardData(portfolio, data, roll_rates=roll_rates)


@pytest.fixture(scope="module")
def latest(dashboard):
    return dashboard.view()


def test_latest_view_is_the_business_date(latest, portfolio):
    today = portfolio[portfolio["business_date"] == pd.Timestamp(AS_OF)]

    assert latest.business_date == pd.Timestamp(AS_OF)
    assert latest.summary["loans"] == len(today)
    assert latest.summary["outstanding"] == int(today["outstanding_balance"].sum())
    assert 0 <= latest.summary["PAR90 %"] <= latest.summary["PAR30 %"] <= 100


def test_first_view_has_no_earlier_period(dashboard):
    assert dashboard.view(dashboard.dates[0]).period_start is None


def test_filters_narrow_the_loans(dashboard, latest):
    view = dashboard.view(profile="high_risk")

    assert set(view.loans["profile"]) == {"high_risk"}
    assert view.summary["loans"] < latest.summary["loans"]


def test_bucket_counts_add_up(latest):
    assert latest.buckets["loans_by_dpd"].sum() == latest.buckets["loans_by_mia"].sum() == latest.summary["loans"]


def test_reachability_bands_cover_every_loan(latest):
    assert latest.reachability["loans"].sum() == latest.summary["loans"]


def test_options_start_with_all(dashboard):
    options = dashboard.options()
    assert options["profile"][0] == options["bucket"][0] == options["tier"][0] == "All"
    assert options["tier"][1:] == ["A", "B", "C", "D", "E"]
    assert options["bucket"][1:3] == ["current", "1-30"]


def test_csv_export_writes_one_file_per_table(tmp_path, latest):
    paths = CsvExporter(tmp_path).export(latest)

    assert len(paths) == len(latest.tables())
    assert all(path.parent == tmp_path / "2026-10-22" and path.stat().st_size for path in paths)


def test_pdf_export(tmp_path, dashboard):
    paths = PdfExporter(tmp_path).export(dashboard.view(profile="high_risk"))

    assert len(paths) == 1
    assert paths[0].name.startswith("dashboard_high_risk_")
    assert paths[0].read_bytes()[:4] == b"%PDF"


def test_unknown_format_and_unwritable_folder(tmp_path, latest):
    with pytest.raises(ExportError, match="Unsupported export format 'XLS'"):
        exporter_for("XLS")
    blocker = tmp_path / "export"
    blocker.write_text("a file where the folder should be")
    with pytest.raises(ExportError, match="Can't create the export folder"):
        CsvExporter(blocker).export(latest)


def test_tiers_and_expected_loss(latest):
    assert latest.tiers["loans"].sum() == latest.summary["loans"]
    assert latest.tiers["expected_loss"].sum() == pytest.approx(latest.summary["expected loss"], abs=5)
    assert latest.loans["expected_loss"].is_monotonic_decreasing


def test_tier_filter(dashboard):
    view = dashboard.view(tier="E")
    assert set(view.loans["tier"]) == {"E"}
    assert view.filters["tier"] == "E"


def test_projection_on_the_latest_date(latest):
    assert list(latest.projection.index) == list(range(13))
    total = latest.summary["outstanding"]
    assert latest.projection.sum(axis=1).to_numpy() == pytest.approx(total)
    assert list(latest.write_offs["months ahead"]) == [3, 6, 12]
    assert latest.write_offs["written off (UGX)"].is_monotonic_increasing


def test_no_projection_for_earlier_dates(dashboard):
    view = dashboard.view(dashboard.dates[-2])
    assert view.projection.empty and view.write_offs.empty
    assert any("only shown for the latest business date" in note for note in view.notes)


def test_no_projection_without_a_model(portfolio, data):
    assert DashboardData(portfolio, data).view().projection.empty


def test_dashboard_renders_as_html(dashboard):
    page = PortfolioDashboard(dashboard)
    html = page.render(page.current_view())

    assert html.count("<img src='data:image/png;base64,") == 1
    assert "Projected new write-offs" in html and "Expected loss (UGX)" in html


def test_dashboard_is_displayed_once(monkeypatch, dashboard):
    shown = []
    monkeypatch.setattr("src.dashboard.portfolio_dashboard.display", shown.append)
    page = PortfolioDashboard(dashboard)

    page.show()
    page.tier.value = "E"

    assert len(shown) == 1
    assert "<img" in page.content.value


def test_dashboard_export_reports_the_files(tmp_path, dashboard):
    page = PortfolioDashboard(dashboard, export_folder=tmp_path)
    page.format.value = "CSV"

    page._export(None)

    assert "Saved 14 file(s)" in page.export_result.value
