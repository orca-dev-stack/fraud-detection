import argparse

from data.unified_loader import load_all
from features.build_features import build_features

from models.train_tabular import train_tabular
from models.pytorch_model import train_pytorch
from models.tensorflow_model import train_tf
from models.autoencoder import train_autoencoder
from models.anomaly_detector import train_anomaly
from models.gnn_blockchain import train_gnn
from models.roc_pr_plots import plot_all



def run_tabular(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    train_tabular(X, y)


def run_pytorch(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    train_pytorch(X, y)


def run_tf(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    train_tf(X, y)


def run_autoencoder(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    train_autoencoder(X, y)


def run_anomaly(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    train_anomaly(X, y)


def run_gnn(df):
    df_crypto = df[df["channel"] == "crypto"]
    if df_crypto.empty:
        print("[GNN] No crypto data, skipping.")
    else:
        train_gnn(df_crypto)


def main():
    parser = argparse.ArgumentParser(description="Osprey unified training CLI")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "tabular", "pytorch", "tf", "autoencoder", "anomaly", "gnn"],
        help="Which model group to train",
    )
    args = parser.parse_args()

    print("\n=== Loading unified dataset ===")
    df = load_all()
    print(f"[Data] Unified dataset: {df.shape}")

    print("\n=== Building features ===")
    df = build_features(df)
    print(f"[Features] Final feature matrix: {df.shape}")

    if args.only in ("all", "tabular"):
        print("\n[Tabular] Training...")
        run_tabular(df)

    if args.only in ("all", "pytorch"):
        print("\n[PyTorch] Training...")
        run_pytorch(df)

    if args.only in ("all", "tf"):
        print("\n[TensorFlow] Training...")
        run_tf(df)

    if args.only in ("all", "autoencoder"):
        print("\n[Autoencoder] Training...")
        run_autoencoder(df)

    if args.only in ("all", "anomaly"):
        print("\n[Anomaly] Training...")
        run_anomaly(df)

    if args.only in ("all", "gnn"):
        print("\n[GNN] Training...")
        run_gnn(df)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()

metrics = plot_all(y_val, preds, model_name="LightGBM_fraud")
print(metrics)
