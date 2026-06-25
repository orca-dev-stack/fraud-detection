import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # fraud-detection/
RAW_DIR = os.path.join(ROOT, "data", "raw")

def ensure_dirs():
    for name in ["ieee", "ccf", "paysim", "banksim", "sfd"]:
        path = os.path.join(RAW_DIR, name)
        os.makedirs(path, exist_ok=True)

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def download_ieee():
    out_dir = os.path.join(RAW_DIR, "ieee")
    run(["kaggle", "competitions", "download", "-c", "ieee-fraud-detection", "-p", out_dir])
    run(["unzip", os.path.join(out_dir, "*.zip"), "-d", out_dir])

def download_ccf():
    out_dir = os.path.join(RAW_DIR, "ccf")
    run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", out_dir])
    run(["unzip", os.path.join(out_dir, "*.zip"), "-d", out_dir])

def download_paysim():
    out_dir = os.path.join(RAW_DIR, "paysim")
    run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1", "-p", out_dir])
    run(["unzip", os.path.join(out_dir, "*.zip"), "-d", out_dir])

def download_banksim():
    out_dir = os.path.join(RAW_DIR, "banksim")
    run(["kaggle", "datasets", "download", "-d", "ntnu-testimon/banksim1", "-p", out_dir])
    run(["unzip", os.path.join(out_dir, "*.zip"), "-d", out_dir])

def download_sfd():
    out_dir = os.path.join(RAW_DIR, "sfd")
    run(["kaggle", "datasets", "download", "-d", "ntnu-testimon/sfd1", "-p", out_dir])
    run(["unzip", os.path.join(out_dir, "*.zip"), "-d", out_dir])

if __name__ == "__main__":
    ensure_dirs()
    download_ieee()
    download_ccf()
    download_paysim()
    download_banksim()
    download_sfd()
