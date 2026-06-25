import pandas as pd
from load_data import load_ieee, load_ccf, load_paysim, load_banksim, load_sfd

def normalize_ieee(df):
    return pd.DataFrame({
        "amount": df["TransactionAmt"],
        "time": df["TransactionDT"],
        "customer_id": df["card1"],
        "merchant_id": df["card2"],
        "transaction_type": df["ProductCD"],
        "isFraud": df["isFraud"],
        "source": "ieee"
    })

def normalize_ccf(df):
    return pd.DataFrame({
        "amount": df["Amount"],
        "time": df["Time"],
        "customer_id": None,
        "merchant_id": None,
        "transaction_type": "card",
        "isFraud": df["Class"],
        "source": "ccf"
    })

def normalize_paysim(df):
    return pd.DataFrame({
        "amount": df["amount"],
        "time": df["step"],
        "customer_id": df["nameOrig"],
        "merchant_id": df["nameDest"],
        "transaction_type": df["type"],
        "isFraud": df["isFraud"],
        "source": "paysim"
    })

def normalize_banksim(df):
    return pd.DataFrame({
        "amount": df["amount"],
        "time": df["timestamp"],
        "customer_id": df["customer"],
        "merchant_id": df["merchant"],
        "transaction_type": df["operation"],
        "isFraud": df["fraud"],
        "source": "banksim"
    })

def normalize_sfd(df):
    return pd.DataFrame({
        "amount": df["amount"],
        "time": df["step"],
        "customer_id": df["nameOrig"],
        "merchant_id": df["nameDest"],
        "transaction_type": df["type"],
        "isFraud": df["isFraud"],
        "source": "sfd"
    })

def merge_all():
    ieee = normalize_ieee(load_ieee())
    ccf = normalize_ccf(load_ccf())
    paysim = normalize_paysim(load_paysim())
    banksim = normalize_banksim(load_banksim())
    sfd = normalize_sfd(load_sfd())

    return pd.concat([ieee, ccf, paysim, banksim, sfd], ignore_index=True)

if __name__ == "__main__":
    df = merge_all()
    print("Merged dataset:", df.shape)
