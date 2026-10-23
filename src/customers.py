import pandas as pd

# how a customer missing from customer.csv is shown, so the row stays visible and easy to spot
MISSING_CUSTOMER = "(customer not found)"


def customer_names(customer_rows: list[dict]) -> pd.DataFrame:
    """Each customer's uuid (as customer_uuid) and name as the lists show it: surname, then given name."""
    customers = pd.DataFrame(customer_rows, columns=["uuid", "surname", "given_name"])
    return pd.DataFrame({"customer_uuid": customers["uuid"],
                         "customer": customers["surname"] + " " + customers["given_name"]})


def add_customer_names(frame: pd.DataFrame, customer_rows: list[dict],
                       missing: str = MISSING_CUSTOMER) -> pd.DataFrame:
    """`frame` with a 'customer' column joined on its customer_uuid; no row is dropped."""
    named = frame.merge(customer_names(customer_rows), how="left", on="customer_uuid")
    named["customer"] = named["customer"].fillna(missing)
    return named
