import csv
from pathlib import Path

from .exceptions import ImportFileError


class CsvImport:
    """
    Reads a CSV file into memory.

    After load(), the column names are in `header` and the data rows are in `rows`.
    Values are kept as text; converting them to numbers or dates is a later step.
    """

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8-sig"):
        """
          utf-8-sig means reads files saved from Excel as well
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.path: Path | None = None
        self.header: list[str] = []
        self.rows: list[list[str]] = []

    def load(self, path: str | Path, include_header: bool = True) -> list[list[str]]:
        """
        Read the CSV file at `path` and return its rows.

        include_header=True returns the header row as the first row.
        include_header=False returns the data rows only.
        Either way, the column names are also kept in `self.header`.
        """
        self.path = self._check_path(path)

        try:
            with self.path.open(newline="", encoding=self.encoding) as file:
                all_rows = list(csv.reader(file, delimiter=self.delimiter))
        except UnicodeDecodeError as error:
            raise ImportFileError(f"{self.path.name} is not valid {self.encoding} text: {error}") from error
        except csv.Error as error:
            raise ImportFileError(f"{self.path.name} is not a valid CSV file: {error}") from error

        # drop completely blank lines
        all_rows = [row for row in all_rows if any(value.strip() for value in row)]
        if not all_rows:
            raise ImportFileError(f"{self.path.name} is empty")

        self.header = [name.strip() for name in all_rows[0]]
        self.rows = all_rows[1:]
        self._check_header()
        self._check_row_lengths()

        return [self.header, *self.rows] if include_header else list(self.rows)

    def load_dir_data(self, folder: str | Path) -> dict[str, dict[str, list]]:
        """
        Load every CSV file in `folder` and its subfolders.
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise ImportFileError(f"Folder not found: {folder_path.resolve()}")

        csv_paths = sorted(folder_path.rglob("*.csv"))
        if not csv_paths:
            raise ImportFileError(f"No CSV files found in {folder_path.resolve()}")

        data = {}
        for path in csv_paths:
            if path.stem in data:
                raise ImportFileError(f"Two files in {folder_path.name} are both called {path.stem}.csv")
            self.load(path)
            data[path.stem] = {
                "header": self.header,
                "rows": self.records(),
            }
        return data

    def records(self) -> list[dict[str, str]]:
        """The data rows as dictionaries keyed by column name, e.g. {'code': 'business', ...}."""
        if self.path is None:
            raise ImportFileError("Nothing loaded yet; call load() first")
        return [dict(zip(self.header, row)) for row in self.rows]

    def _check_path(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.exists():
            raise ImportFileError(f"File not found: {path}")
        if not path.is_file():
            raise ImportFileError(f"Not a file: {path}")
        if path.suffix.lower() != ".csv":
            raise ImportFileError(f"Expected a .csv file, got: {path.name}")
        return path

    def _check_header(self) -> None:
        if any(name == "" for name in self.header):
            raise ImportFileError(f"{self.path.name} has a blank column name in its header row")
        duplicates = sorted({name for name in self.header if self.header.count(name) > 1})
        if duplicates:
            raise ImportFileError(f"{self.path.name} has duplicate column names: {', '.join(duplicates)}")

    def _check_row_lengths(self) -> None:
        """
          line 1 is the header, so the first data row is line 2
        """
        for line_number, row in enumerate(self.rows, start=2):
            if len(row) != len(self.header):
                raise ImportFileError(
                    f"{self.path.name}, row {line_number}: expected {len(self.header)} values, found {len(row)}"
                )