import base64
import html
import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from ..exceptions import SaccoError
from .charts import dashboard_figure
from .dashboard_data import ALL, DashboardData
from .report_exporter import EXPORTERS, exporter_for


class PortfolioDashboard:
    """
    The collections dashboard: filters on top, then headline figures, charts and tables,
    and an export bar that saves the current view as CSV or PDF.

    Everything below the filters is HTML held in widgets and replaced on each change, rather than
    display() calls captured by Output widgets, so it renders once in every notebook frontend.
    """

    def __init__(self, data: DashboardData, export_folder: str = "export"):
        self.data = data
        self.export_folder = export_folder
        options = data.options()
        style = {"description_width": "110px"}
        self.business_date = widgets.Dropdown(
            description="Business date", options=[(f"{d:%Y-%m-%d}", d) for d in options["business_date"]],
            value=options["business_date"][-1], style=style)
        self.assigned_to = widgets.Dropdown(description="Officer", options=options["assigned_to"], value=ALL, style=style)
        self.profile = widgets.Dropdown(description="Profile", options=options["profile"], value=ALL, style=style)
        self.bucket = widgets.Dropdown(description="Bucket", options=options["bucket"], value=ALL, style=style)
        self.tier = widgets.Dropdown(description="Risk tier", options=options["tier"], value=ALL, style=style)
        self.format = widgets.Dropdown(description="Format", options=list(EXPORTERS), value="PDF", style=style)
        self.export_button = widgets.Button(description="Export", icon="download", button_style="primary")
        self.export_result = widgets.HTML()
        self.content = widgets.HTML()

        for widget in (self.business_date, self.assigned_to, self.profile, self.bucket, self.tier):
            widget.observe(self._refresh, names="value")
        self.export_button.on_click(self._export)

    def current_view(self):
        return self.data.view(self.business_date.value, self.assigned_to.value, self.profile.value,
                              self.bucket.value, self.tier.value)

    def show(self) -> None:
        filters = widgets.GridBox([self.business_date, self.assigned_to, self.profile, self.bucket, self.tier],
                                  layout=widgets.Layout(grid_template_columns="repeat(3, 1fr)", grid_gap="8px 24px"))
        export_bar = widgets.HBox([self.format, self.export_button])
        panel = widgets.VBox([widgets.HTML("<h3 style='margin:0 0 8px 0'>Collections dashboard</h3>"), filters,
                              export_bar, self.export_result],
                             layout=widgets.Layout(border="1px solid #d0d7de", padding="12px 16px"))
        display(widgets.VBox([panel, self.content]))
        self._refresh()

    def _refresh(self, *_):
        self.content.value = self.render(self.current_view())

    def render(self, view) -> str:
        """The whole dashboard below the filters, as one HTML string."""
        parts = [self._cards(view.summary), self._figure_html(view)]
        parts += [f"<p>{html.escape(note)}</p>" for note in view.notes]
        tables = [("Risk tiers", view.tiers), ("Profiles", view.profiles), ("Buckets by DPD and MIA", view.buckets),
                  ("Projected new write-offs", view.write_offs),
                  ("Contact outcomes in the period", view.outcomes),
                  ("Treatments blocked by each compliance rule", view.compliance),
                  ("Payments received in the period", view.payments),
                  ("Promises due in the period", view.promises),
                  ("Loans, highest expected loss first (top 25)", view.loans.head(25))]
        for title, table in tables:
            parts.append(f"<h4 style='margin:12px 0 4px 0'>{title}</h4>")
            parts.append(table.style.hide(axis="index").format(precision=2, thousands=",").to_html() if len(table)
                         else "<i>nothing to show for these filters</i>")
        return "".join(parts)

    # an ordinary method, not a staticmethod: %autoreload drops the staticmethod wrapper when it adds
    # a new method to a class already loaded in the kernel
    def _figure_html(self, view) -> str:
        figure = dashboard_figure(view)
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=80)
        plt.close(figure)
        encoded = base64.b64encode(image.getvalue()).decode("ascii")
        return f"<img src='data:image/png;base64,{encoded}' style='max-width:100%'>"

    @staticmethod
    def _cards(summary: dict) -> str:
        cards = [("Loans", f"{summary['loans']:,}"), ("Outstanding (UGX)", f"{summary['outstanding']:,}"),
                 ("PAR30", f"{summary['PAR30 %']}%"), ("PAR90", f"{summary['PAR90 %']}%"),
                 ("Average PD", f"{summary['average PD %']}%"),
                 ("Expected loss (UGX)", f"{summary['expected loss']:,}"),
                 ("Payments in period (UGX)", f"{summary['payments received']:,}"),
                 ("Promise keep rate", f"{summary['promise keep rate %']}%")]
        boxes = "".join(f"<div style='border:1px solid #d0d7de;padding:8px 12px;margin:4px;min-width:120px'>"
                        f"<div style='font-size:0.8em;color:#57606a'>{label}</div><div style='font-size:1.3em'>"
                        f"<b>{value}</b></div></div>" for label, value in cards)
        return f"<div style='display:flex;flex-wrap:wrap'>{boxes}</div>"

    def _export(self, _):
        try:
            paths = exporter_for(self.format.value, self.export_folder).export(self.current_view())
        except SaccoError as error:
            self.export_result.value = f"<span style='color:#cf222e'>Export failed: {html.escape(str(error))}</span>"
            return
        links = "".join(f"<li><a href='{html.escape(path.as_posix())}' target='_blank'>{html.escape(path.as_posix())}</a></li>"
                        for path in paths)
        self.export_result.value = f"Saved {len(paths)} file(s):<ul style='margin:4px 0'>{links}</ul>"
