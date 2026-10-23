import pytest

from src.csv_import import CsvImport
from src.exceptions import ImportFileError, SaccoError


def write(path, text, encoding="utf-8"):
    path.write_text(text, encoding=encoding)
    return path


def test_load_returns_header_and_rows(tmp_path):
    path = write(tmp_path / "people.csv", "code,name\nA,Alice\nB,Bob\n")
    importer = CsvImport()

    assert importer.load(path) == [["code", "name"], ["A", "Alice"], ["B", "Bob"]]
    assert importer.load(path, include_header=False) == [["A", "Alice"], ["B", "Bob"]]
    assert importer.header == ["code", "name"]


def test_records_are_keyed_by_column(tmp_path):
    importer = CsvImport()
    importer.load(write(tmp_path / "people.csv", "code,name\nA,Alice\n"))

    assert importer.records() == [{"code": "A", "name": "Alice"}]


def test_blank_lines_and_header_spaces_are_ignored(tmp_path):
    importer = CsvImport()
    importer.load(write(tmp_path / "people.csv", " code , name \n\nA,Alice\n , \n"))

    assert importer.header == ["code", "name"]
    assert importer.rows == [["A", "Alice"]]


def test_excel_byte_order_mark_is_removed(tmp_path):
    importer = CsvImport()
    importer.load(write(tmp_path / "people.csv", "code,name\nA,Alice\n", encoding="utf-8-sig"))

    assert importer.header[0] == "code"


def test_quoted_values_keep_their_commas(tmp_path):
    importer = CsvImport()
    importer.load(write(tmp_path / "notes.csv", 'id,note\n1,"Plot 5, Jinja Road"\n'))

    assert importer.records()[0]["note"] == "Plot 5, Jinja Road"


@pytest.mark.parametrize("text, message", [
    ("", "is empty"),
    ("\n\n", "is empty"),
    ("code,,name\nA,x,Alice\n", "blank column name"),
    ("code,code\nA,B\n", "duplicate column names: code"),
    ("code,name\nA\n", "row 2: expected 2 values, found 1"),
])
def test_badly_formed_files_are_rejected(tmp_path, text, message):
    with pytest.raises(ImportFileError, match=message):
        CsvImport().load(write(tmp_path / "bad.csv", text))


def test_file_that_is_not_utf8_is_rejected(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name\nMüller\n".encode("latin-1"))

    with pytest.raises(ImportFileError, match="not valid utf-8-sig"):
        CsvImport().load(path)


def test_missing_file_folder_and_wrong_extension(tmp_path):
    importer = CsvImport()
    with pytest.raises(ImportFileError, match="File not found"):
        importer.load(tmp_path / "missing.csv")
    with pytest.raises(ImportFileError, match="Not a file"):
        importer.load(tmp_path)
    with pytest.raises(ImportFileError, match="Expected a .csv file"):
        importer.load(write(tmp_path / "people.txt", "code\nA\n"))


def test_records_before_load():
    with pytest.raises(ImportFileError, match="call load"):
        CsvImport().records()


def test_import_errors_share_the_sacco_base_class():
    assert issubclass(ImportFileError, SaccoError)


def test_load_dir_data_reads_subfolders(tmp_path):
    write(tmp_path / "a.csv", "x\n1\n")
    (tmp_path / "more").mkdir()
    write(tmp_path / "more" / "b.csv", "y\n2\n3\n")

    data = CsvImport().load_dir_data(tmp_path)

    assert data == {"a": {"header": ["x"], "rows": [{"x": "1"}]},
                    "b": {"header": ["y"], "rows": [{"y": "2"}, {"y": "3"}]}}


def test_load_dir_data_rejects_two_files_with_the_same_name(tmp_path):
    write(tmp_path / "a.csv", "x\n1\n")
    (tmp_path / "more").mkdir()
    write(tmp_path / "more" / "a.csv", "x\n2\n")

    with pytest.raises(ImportFileError, match="both called a.csv"):
        CsvImport().load_dir_data(tmp_path)


def test_load_dir_data_needs_a_folder_with_csv_files(tmp_path):
    with pytest.raises(ImportFileError, match="Folder not found"):
        CsvImport().load_dir_data(tmp_path / "missing")
    with pytest.raises(ImportFileError, match="No CSV files"):
        CsvImport().load_dir_data(tmp_path)


def test_project_data_loads(data, kb):
    assert {"customer", "case", "case_snapshot", "contact_history", "payment_history", "ptp",
            "treatment_history"} <= set(data)
    assert {"loan_types", "locations", "treatment_type"} <= set(kb)
