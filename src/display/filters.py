from abc import ABC, abstractmethod

import ipywidgets as widgets
import pandas as pd

from ..exceptions import DisplayError


class Filter(ABC):
    """
    One filter on a list, e.g. a search box or a dropdown.

    Each filter owns its widget and knows which rows of a table it keeps.
    """

    def __init__(self, column: str, label: str, width: str = "100%"):
        self.column = column
        self.label = label
        self.width = width
        self.widget: widgets.Widget | None = None

    def build(self, frame: pd.DataFrame) -> widgets.Widget:
        """Create the widget, using the table to set things like dropdown options."""
        if self.column not in frame.columns:
            raise DisplayError(f"Filter '{self.label}' refers to unknown column '{self.column}'")
        self.widget = self._make_widget(frame[self.column])
        return self.widget

    @abstractmethod
    def _make_widget(self, values: pd.Series) -> widgets.Widget:
        """Create the widget for this kind of filter."""

    @abstractmethod
    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """True for each row the filter keeps."""

    @abstractmethod
    def reset(self) -> None:
        """Put the widget back to 'show everything'."""


class TextFilter(Filter):
    """Keeps rows whose column contains the typed text, ignoring case."""

    def __init__(self, column: str, label: str, max_length: int | None = None, width: str = "100%"):
        super().__init__(column, label, width)
        self.max_length = max_length

    def _make_widget(self, values):
        box = widgets.Text(description=self.label, placeholder="type to search")
        if self.max_length:
            box.observe(self._trim, names="value")
        return box

    def _trim(self, change):
        if len(change["new"]) > self.max_length:
            self.widget.value = change["new"][: self.max_length]

    def mask(self, frame):
        text = self.widget.value.strip()
        if not text:
            return pd.Series(True, index=frame.index)
        return frame[self.column].astype(str).str.contains(text, case=False, regex=False)

    def reset(self):
        self.widget.value = ""


class ChoiceFilter(Filter):
    """A dropdown of the column's values, plus 'All'."""

    ALL = "All"

    def _make_widget(self, values):
        options = [self.ALL, *sorted(values.dropna().astype(str).unique())]
        return widgets.Dropdown(description=self.label, options=options)

    def mask(self, frame):
        if self.widget.value == self.ALL:
            return pd.Series(True, index=frame.index)
        return frame[self.column].astype(str) == self.widget.value

    def reset(self):
        self.widget.value = self.ALL


class RangeFilter(Filter):
    """A slider keeping rows whose number lies between the two handles."""

    def __init__(self, column: str, label: str, step: int = 1, width: str = "100%"):
        super().__init__(column, label, width)
        self.step = step

    def _make_widget(self, values):
        numbers = pd.to_numeric(values, errors="raise")
        low, high = int(numbers.min()), int(numbers.max())
        return widgets.IntRangeSlider(description=self.label, min=low, max=high,
                                      value=(low, high), step=self.step)

    def mask(self, frame):
        low, high = self.widget.value
        return pd.to_numeric(frame[self.column]).between(low, high)

    def reset(self):
        self.widget.value = (self.widget.min, self.widget.max)