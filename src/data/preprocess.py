import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df):
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])
    df["is_large"] = (df["amount"] > df["amount"].median()).astype(int)

    df["hour"] = df["time"] % 24

    df["customer_freq"] = df.groupby("customer_id")["amount"].transform("count")
    df["customer_avg_amt"] = df.groupby("customer_id")["amount"].transform("mean")

    df["merchant_freq"] = df.groupby("merchant_id")["amount"].transform("count")

    df = df.fillna(-1)

    X = df.drop(["isFraud"], axis=1)
    y = df["isFraud"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_valid, y_train, y_valid

if __name__ == "__main__":
    from merge_data import merge_all
    df = merge_all()
    X_train, X_valid, y_train, y_valid = preprocess(df)
    print("Train:", X_train.shape, "Valid:", X_valid.shape)
