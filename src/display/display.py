import math
from abc import ABC, abstractmethod

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from ..exceptions import DisplayError
from .filters import Filter


class Display(ABC):
    """
    A searchable list shown in the notebook: filters on top, matching rows below.

    Subclasses decide what goes in the table and which filters it has, by overriding
    build_frame() and build_filters(), and setting title, columns and sort_by.
    Display does the rest: building the widgets, filtering, paging and redrawing.

    page_size and filter_cols_per_row can be passed when creating a list.
    """

    title: str = "List"
    columns: list[str] = []
    sort_by: list[str] = []
    sort_ascending: bool | list[bool] = True
    label_width = "90px"
    align_numbers_right = True     # text is left-aligned; numbers stay right-aligned so digits line up

    # left-aligns the filter labels; only applies inside the filter grid
    FILTER_STYLE = "<style>.sacco-filters .widget-label { text-align: left !important; }</style>"

    def __init__(self, page_size: int = 50, filter_cols_per_row: int = 3):
        if page_size < 1:
            raise DisplayError(f"page_size must be at least 1, not {page_size}")
        if filter_cols_per_row < 1:
            raise DisplayError(f"filter_cols_per_row must be at least 1, not {filter_cols_per_row}")
        self.page_size = page_size
        self.filter_cols_per_row = filter_cols_per_row
        self.page = 1                      # the page currently shown, counting from 1

        self.frame = self.build_frame()
        self.filters = self.build_filters()
        missing = [column for column in self.columns + self.sort_by if column not in self.frame.columns]
        if missing:
            raise DisplayError(f"{type(self).__name__} has unknown columns: {', '.join(missing)}")

        # build and wire up the widgets once, so showing the list twice doesn't double the work
        self._widgets = [f.build(self.frame) for f in self.filters]
        for f, widget in zip(self.filters, self._widgets):
            widget.style.description_width = self.label_width
            widget.layout.width = f.width
            widget.observe(self._on_filter_change, names="value")
        self._count = widgets.HTML()
        # the table is HTML held in a widget, not display() calls captured by an Output widget:
        # how Output captures depends on the frontend, and VS Code can show the captured table twice
        self._table = widgets.HTML(layout=widgets.Layout(overflow="auto"))

        # one set of paging controls above the table and one below
        self._top_pager = self._make_pager()
        self._bottom_pager = self._make_pager()

    @abstractmethod
    def build_frame(self) -> pd.DataFrame:
        """Return the full table this list shows and filters."""

    @abstractmethod
    def build_filters(self) -> list[Filter]:
        """Return the filters for this list."""

    def matches(self) -> pd.DataFrame:
        """The rows that pass every filter, sorted, with only the shown columns."""
        keep = pd.Series(True, index=self.frame.index)
        for f in self.filters:
            keep &= f.mask(self.frame)
        result = self.frame[keep]
        if self.sort_by:
            result = result.sort_values(self.sort_by, ascending=self.sort_ascending)

        result = result[self.columns].reset_index(drop=True)
        result.index += 1          # number the rows from 1 instead of 0
        return result

    def page_count(self, matches: pd.DataFrame | None = None) -> int:
        """How many pages the matching rows fill; at least 1, even when nothing matches."""
        rows = len(self.matches() if matches is None else matches)
        return max(1, math.ceil(rows / self.page_size))

    def current_page(self) -> pd.DataFrame:
        """The matching rows on the page currently shown."""
        start = (self.page - 1) * self.page_size
        return self.matches().iloc[start:start + self.page_size]

    def go_to_page(self, page: int) -> None:
        """Show the given page, kept within the first and last pages."""
        self.page = min(max(1, page), self.page_count())
        self._refresh()

    def _make_pager(self) -> dict[str, widgets.Widget]:
        """A set of paging buttons and a page label, wired to this list."""
        button = widgets.Layout(width="auto")
        pager = {
            "first": widgets.Button(description="First", icon="angle-double-left", layout=button),
            "previous": widgets.Button(description="Previous", icon="angle-left", layout=button),
            "label": widgets.HTML(),
            "next": widgets.Button(description="Next", icon="angle-right", layout=button),
            "last": widgets.Button(description="Last", icon="angle-double-right", layout=button),
        }
        pager["first"].on_click(lambda _: self.go_to_page(1))
        pager["previous"].on_click(lambda _: self.go_to_page(self.page - 1))
        pager["next"].on_click(lambda _: self.go_to_page(self.page + 1))
        pager["last"].on_click(lambda _: self.go_to_page(self.page_count()))
        return pager

    @staticmethod
    def _pager_row(pager: dict[str, widgets.Widget], margin: str) -> widgets.HBox:
        return widgets.HBox(list(pager.values()), layout=widgets.Layout(
            justify_content="center", align_items="center", grid_gap="8px", margin=margin))

    def _styled(self, rows: pd.DataFrame):
        """The rows ready to display: headers and text left-aligned, numbers optionally right-aligned."""
        styled = (rows.style
                  .set_properties(**{"text-align": "left"})
                  .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))
        if self.align_numbers_right:
            numbers = list(rows.select_dtypes("number").columns)
            if numbers:
                styled = styled.set_properties(subset=numbers, **{"text-align": "right"})
        return styled

    def show(self) -> None:
        """Draw the filters, the table and the paging controls in the notebook."""
        filters = widgets.GridBox(
            self._widgets,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({self.filter_cols_per_row}, 1fr)",
                grid_gap="8px 24px",              # space between rows, space between columns
            ),
        )
        filters.add_class("sacco-filters")

        clear = widgets.Button(description="Clear filters", icon="times",
                               layout=widgets.Layout(width="auto"))
        clear.on_click(self._reset)

        toolbar = widgets.HBox(
            [self._count, clear],
            layout=widgets.Layout(justify_content="space-between", align_items="center",
                                  margin="8px 0 0 0"),
        )

        panel = widgets.VBox(
            [widgets.HTML(self.FILTER_STYLE),
             widgets.HTML(f"<h3 style='margin:0 0 8px 0'>{self.title}</h3>"), filters, toolbar],
            layout=widgets.Layout(border="1px solid #d0d7de", padding="12px 16px",
                                  margin="0 0 12px 0"),
        )

        display(widgets.VBox([
            panel,
            self._pager_row(self._top_pager, margin="0 0 8px 0"),
            self._table,
            self._pager_row(self._bottom_pager, margin="8px 0 0 0"),
        ]))
        self._refresh()

    def _refresh(self, *_):
        result = self.matches()
        pages = self.page_count(result)
        self.page = min(self.page, pages)        # the page may no longer exist after filtering

        start = (self.page - 1) * self.page_size
        page_rows = result.iloc[start:start + self.page_size]

        if result.empty:
            self._count.value = f"<b>0</b> of {len(self.frame)} {self.title.lower()}"
        else:
            self._count.value = (f"Showing <b>{start + 1}–{start + len(page_rows)}</b> of {len(result)}"
                                 f" matching ({len(self.frame)} {self.title.lower()} in total)")
        on_first, on_last = self.page == 1, self.page == pages
        for pager in (self._top_pager, self._bottom_pager):
            pager["label"].value = f"Page <b>{self.page}</b> of {pages}"
            pager["first"].disabled = pager["previous"].disabled = on_first
            pager["next"].disabled = pager["last"].disabled = on_last

        self._table.value = self._styled(page_rows).to_html()

    def _on_filter_change(self, *_):
        # a new search starts again from the first page
        self.page = 1
        self._refresh()

    def _reset(self, _):
        for f in self.filters:
            f.reset()
        self.page = 1
        self._refresh()