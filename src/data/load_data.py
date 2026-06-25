import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

def load_ieee():
    path = os.path.join(RAW, "ieee")
    trans = pd.read_csv(os.path.join(path, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(path, "train_identity.csv"))
    return trans.merge(ident, on="TransactionID", how="left")

def load_ccf():
    path = os.path.join(RAW, "ccf")
    return pd.read_csv(os.path.join(path, "creditcard.csv"))

def load_paysim():
    path = os.path.join(RAW, "paysim")
    return pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"))

def load_banksim():
    path = os.path.join(RAW, "banksim")
    return pd.read_csv(os.path.join(path, "bs140513_032310.csv"))

def load_sfd():
    path = os.path.join(RAW, "sfd")
    return pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"))

if __name__ == "__main__":
    print("IEEE:", load_ieee().shape)
    print("CCF:", load_ccf().shape)
    print("PaySim:", load_paysim().shape)
    print("BankSim:", load_banksim().shape)
    print("SFD:", load_sfd().shape)
