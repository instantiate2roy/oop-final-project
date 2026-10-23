"""The Display base class and its filters, tested through a small list built for the tests."""
import pandas as pd
import pytest

from src.display.display import Display
from src.display.filters import ChoiceFilter, RangeFilter, TextFilter
from src.exceptions import DisplayError

ROWS = [
    {"name": "Apio", "kind": "fruit", "value": 30},
    {"name": "abc", "kind": "letters", "value": 10},
    {"name": "a.c", "kind": "letters", "value": 20},
    {"name": "Okot", "kind": "fruit", "value": 50},
    {"name": "Kato", "kind": "fruit", "value": 40},
]


class NumberList(Display):
    title = "Numbers"
    columns = ["name", "kind", "value"]
    sort_by = ["value"]

    def __init__(self, rows=ROWS, **options):
        self.rows = rows
        super().__init__(**options)

    def build_frame(self):
        return pd.DataFrame(self.rows)

    def build_filters(self):
        return [TextFilter("name", "Name"), ChoiceFilter("kind", "Kind"), RangeFilter("value", "Value")]


def names(frame):
    return list(frame["name"])


def test_display_is_abstract():
    with pytest.raises(TypeError):
        Display()


def test_matches_are_sorted_and_numbered_from_one():
    matches = NumberList().matches()

    assert list(matches["value"]) == [10, 20, 30, 40, 50]
    assert list(matches.index) == [1, 2, 3, 4, 5]
    assert list(matches.columns) == NumberList.columns


def test_text_filter_ignores_case_and_treats_text_literally():
    numbers = NumberList()
    text = numbers.filters[0]

    text.widget.value = "A"
    assert names(numbers.matches()) == ["abc", "a.c", "Apio", "Kato"]
    text.widget.value = "a.c"                       # "." is a dot, not "any character"
    assert names(numbers.matches()) == ["a.c"]


def test_text_filter_max_length_trims_typing():
    text = TextFilter("name", "Name", max_length=3)
    text.build(pd.DataFrame(ROWS))

    text.widget.value = "Okoth"
    assert text.widget.value == "Oko"


def test_choice_filter_lists_values_and_all():
    numbers = NumberList()
    choice = numbers.filters[1]

    assert list(choice.widget.options) == ["All", "fruit", "letters"]
    choice.widget.value = "letters"
    assert names(numbers.matches()) == ["abc", "a.c"]


def test_range_filter_keeps_values_between_handles():
    numbers = NumberList()
    value = numbers.filters[2]

    assert (value.widget.min, value.widget.max) == (10, 50)
    value.widget.value = (20, 40)
    assert names(numbers.matches()) == ["a.c", "Apio", "Kato"]


def test_filters_combine():
    numbers = NumberList()
    numbers.filters[1].widget.value = "fruit"
    numbers.filters[2].widget.value = (35, 50)

    assert names(numbers.matches()) == ["Kato", "Okot"]


def test_reset_clears_every_filter():
    numbers = NumberList()
    numbers.filters[0].widget.value = "zzz"
    numbers.filters[1].widget.value = "fruit"
    numbers.filters[2].widget.value = (20, 20)

    numbers._reset(None)

    assert len(numbers.matches()) == len(ROWS)


def test_paging():
    numbers = NumberList(page_size=2)

    assert numbers.page_count() == 3
    assert names(numbers.current_page()) == ["abc", "a.c"]
    numbers.go_to_page(3)
    assert names(numbers.current_page()) == ["Okot"]
    numbers.go_to_page(99)
    assert numbers.page == 3
    numbers.go_to_page(-5)
    assert numbers.page == 1


def test_new_search_starts_from_the_first_page():
    numbers = NumberList(page_size=2)
    numbers.go_to_page(3)

    numbers.filters[1].widget.value = "fruit"

    assert numbers.page == 1


def test_no_matches_still_has_one_page():
    numbers = NumberList()
    numbers.filters[0].widget.value = "nothing like this"

    assert numbers.matches().empty
    assert numbers.page_count() == 1


@pytest.mark.parametrize("options, message", [
    ({"page_size": 0}, "page_size must be at least 1"),
    ({"filter_cols_per_row": 0}, "filter_cols_per_row must be at least 1"),
])
def test_bad_options(options, message):
    with pytest.raises(DisplayError, match=message):
        NumberList(**options)


def test_unknown_columns_are_reported():
    class Broken(NumberList):
        columns = ["name", "colour"]

    with pytest.raises(DisplayError, match="unknown columns: colour"):
        Broken()


def test_filter_on_unknown_column_is_reported():
    with pytest.raises(DisplayError, match="unknown column 'colour'"):
        ChoiceFilter("colour", "Colour").build(pd.DataFrame(ROWS))


def test_styled_rows_render_to_html():
    html = NumberList()._styled(NumberList().current_page()).to_html()
    assert "Apio" in html


def test_list_is_displayed_once_and_redrawn_in_place(monkeypatch):
    shown = []
    monkeypatch.setattr("src.display.display.display", shown.append)
    numbers = NumberList(page_size=2)

    numbers.show()
    assert len(shown) == 1                          # the whole list, table included, in one output
    assert numbers._table.value.count("<table") == 1
    assert "abc" in numbers._table.value

    numbers.filters[1].widget.value = "fruit"
    numbers.go_to_page(2)
    assert len(shown) == 1                          # filtering and paging redraw the table, they don't add outputs
    assert "Okot" in numbers._table.value and "abc" not in numbers._table.value
