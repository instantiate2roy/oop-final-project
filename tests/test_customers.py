import pandas as pd
import pytest

from src.customers import MISSING_CUSTOMER, add_customer_names, customer_names
from src.display.contact_history import ContactHistory
from src.display.history_list import HistoryList
from src.display.payment_history import PaymentHistory
from src.display.ptp_history import PtpHistory
from src.display.treatment_history import TreatmentHistory

CUSTOMERS = [{"uuid": "c1", "surname": "Okot", "given_name": "Peter", "sex": "M"},
             {"uuid": "c2", "surname": "Apio", "given_name": "Grace", "sex": "F"}]


def test_customer_names_are_surname_then_given_name():
    names = customer_names(CUSTOMERS)
    assert names.to_dict("records") == [{"customer_uuid": "c1", "customer": "Okot Peter"},
                                        {"customer_uuid": "c2", "customer": "Apio Grace"}]


def test_add_customer_names_keeps_every_row():
    frame = pd.DataFrame({"case_uuid": ["k1", "k2", "k3"], "customer_uuid": ["c2", "gone", "c1"]})

    named = add_customer_names(frame, CUSTOMERS)

    assert list(named["customer"]) == ["Apio Grace", MISSING_CUSTOMER, "Okot Peter"]
    assert list(named["case_uuid"]) == ["k1", "k2", "k3"]          # order kept, nothing dropped


def test_add_customer_names_with_no_customers():
    named = add_customer_names(pd.DataFrame({"customer_uuid": ["c1"]}), [], missing="?")
    assert list(named["customer"]) == ["?"]


def test_history_list_is_an_abstract_base():
    with pytest.raises(TypeError):
        HistoryList([], [])


@pytest.mark.parametrize("history", [ContactHistory, PaymentHistory, PtpHistory, TreatmentHistory])
def test_every_history_list_shares_the_base(history):
    assert issubclass(history, HistoryList)
    assert history.MISSING_CUSTOMER == MISSING_CUSTOMER
    assert history.noun != HistoryList.noun
