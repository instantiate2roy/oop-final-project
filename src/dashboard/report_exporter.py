from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ..exceptions import ExportError
from .charts import dashboard_figure
from .dashboard_data import DashboardView


class ReportExporter(ABC):
    """Saves a dashboard view to files in a folder named after its business date."""

    extension = ""

    def __init__(self, folder: str | Path = "export"):
        self.folder = Path(folder)

    @abstractmethod
    def export(self, view: DashboardView) -> list[Path]:
        """Write the view and return the files created."""

    def _target(self, view: DashboardView, name: str) -> Path:
        """export/<date>/<name>_<filters>_<time>.<ext>; the time stops a file being overwritten."""
        folder = self.folder / f"{view.business_date:%Y-%m-%d}"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExportError(f"Can't create the export folder {folder}: {error}") from error
        filters = "_".join(v.replace(" ", "-") for v in view.filters.values() if v != "All") or "all"
        return folder / f"{name}_{filters}_{datetime.now():%H%M%S}.{self.extension}"


class CsvExporter(ReportExporter):
    """One CSV file per dashboard table."""

    extension = "csv"

    def export(self, view: DashboardView) -> list[Path]:
        paths = []
        for name, table in view.tables().items():
            path = self._target(view, name)
            table.to_csv(path, index=False)
            paths.append(path)
        return paths


class PdfExporter(ReportExporter):
    """A PDF report: the charts on the first page, then the main tables."""

    extension = "pdf"
    ROWS_PER_PAGE = 30

    def export(self, view: DashboardView) -> list[Path]:
        path = self._target(view, "dashboard")
        with PdfPages(path) as pdf:
            figure = dashboard_figure(view)
            pdf.savefig(figure)
            plt.close(figure)
            for title, table in [("Summary", pd.DataFrame([view.summary]).T.reset_index()
                                  .rename(columns={"index": "measure", 0: "value"})),
                                 ("Risk tiers", view.tiers), ("Profiles", view.profiles), ("Buckets", view.buckets),
                                 ("Projected new write-offs", view.write_offs),
                                 ("Contact outcomes", view.outcomes), ("Compliance blocks", view.compliance),
                                 ("Payments", view.payments), ("Promises", view.promises),
                                 ("Loans (highest expected loss first)", view.loans)]:
                self._table_pages(pdf, title, table, view)
        return [path]

    def _table_pages(self, pdf: PdfPages, title: str, table: pd.DataFrame, view: DashboardView) -> None:
        if table.empty:
            table = pd.DataFrame({"note": ["nothing to show for these filters"]})
        table = table.copy()
        for column in table.select_dtypes("float").columns:
            table[column] = table[column].round(3)
        for start in range(0, len(table), self.ROWS_PER_PAGE):
            chunk = table.iloc[start:start + self.ROWS_PER_PAGE]
            figure, axis = plt.subplots(figsize=(11.7, 8.3))      # A4 landscape
            axis.axis("off")
            page = f" (rows {start + 1}-{start + len(chunk)} of {len(table)})" if len(table) > self.ROWS_PER_PAGE else ""
            axis.set_title(f"{title}{page} - business date {view.business_date:%Y-%m-%d}", loc="left")
            cells = axis.table(cellText=chunk.astype(str).values, colLabels=list(chunk.columns), loc="upper left")
            cells.auto_set_font_size(False)
            cells.set_fontsize(7)
            pdf.savefig(figure)
            plt.close(figure)


EXPORTERS = {"CSV": CsvExporter, "PDF": PdfExporter}


def exporter_for(format_name: str, folder: str | Path = "export") -> ReportExporter:
    if format_name not in EXPORTERS:
        raise ExportError(f"Unsupported export format '{format_name}'; choose one of {', '.join(EXPORTERS)}")
    return EXPORTERS[format_name](folder)
