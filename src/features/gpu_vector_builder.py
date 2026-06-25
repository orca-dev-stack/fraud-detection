import cudf
import cupy as cp
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("gpu_vector_builder")


# ---------------------------------------------------------
# 1. Amount transformations (GPU)
# ---------------------------------------------------------

def gpu_transform_amount(amount_col, user_avg_col=None):
    log_amount = cp.log1p(amount_col)

    if user_avg_col is not None:
        ratio = amount_col / cp.maximum(user_avg_col, 1e-6)
    else:
        ratio = cp.ones_like(amount_col)

    return cudf.DataFrame({
        "log_amount": log_amount,
        "amount_ratio": ratio,
    })


# ---------------------------------------------------------
# 2. Time features (GPU)
# ---------------------------------------------------------

def gpu_time_features(ts_col, last_ts_col=None):
    ts = cudf.to_datetime(ts_col)

    hour = ts.dt.hour
    day_of_week = ts.dt.weekday

    if last_ts_col is not None:
        last_ts = cudf.to_datetime(last_ts_col)
        delta = (ts - last_ts).dt.total_seconds()
    else:
        delta = cp.full(len(ts), 999999)

    return cudf.DataFrame({
        "hour": hour,
        "day_of_week": day_of_week,
        "time_since_last_tx": delta,
    })


# ---------------------------------------------------------
# 3. Velocity features (GPU)
# ---------------------------------------------------------

def gpu_velocity_features(tx_times_col):
    ts = cudf.to_datetime(tx_times_col)
    diffs = (ts.max() - ts.min()).total_seconds()

    if diffs == 0:
        per_min = per_hour = per_day = len(ts)
    else:
        per_min = len(ts) / (diffs / 60)
        per_hour = len(ts) / (diffs / 3600)
        per_day = len(ts) / (diffs / 86400)

    return cudf.DataFrame({
        "tx_per_min": [per_min],
        "tx_per_hour": [per_hour],
        "tx_per_day": [per_day],
    })


# ---------------------------------------------------------
# 4. Device features (GPU)
# ---------------------------------------------------------

def gpu_device_features(device_id_col, user_devices_col, device_users_col):
    return cudf.DataFrame({
        "device_seen_before": (device_id_col.isin(user_devices_col)).astype("int8"),
        "device_user_count": device_users_col.astype("int32"),
        "user_device_count": user_devices_col.astype("int32"),
    })


# ---------------------------------------------------------
# 5. IP features (GPU)
# ---------------------------------------------------------

def gpu_ip_features(ip_asn_col, ip_proxy_col, ip_risk_col):
    return cudf.DataFrame({
        "ip_asn": ip_asn_col.astype("int32"),
        "ip_is_proxy": ip_proxy_col.astype("int8"),
        "ip_risk_score": ip_risk_col.astype("float32"),
    })


# ---------------------------------------------------------
# 6. Behavioral features (GPU)
# ---------------------------------------------------------

def gpu_behavioral_features(df):
    cols = [
        "avg_amount", "std_amount",
        "merchant_diversity", "country_diversity",
        "device_switch_rate"
    ]
    return df[cols].astype("float32")


# ---------------------------------------------------------
# 7. Risk features (GPU)
# ---------------------------------------------------------

def gpu_risk_features(news_col, market_col, mempool_col):
    return cudf.DataFrame({
        "news_risk": news_col.astype("float32"),
        "market_volatility": market_col.astype("float32"),
        "mempool_pressure": mempool_col.astype("float32"),
    })


# ---------------------------------------------------------
# 8. Graph features (GPU)
# ---------------------------------------------------------

def gpu_graph_features(df):
    cols = ["graph_degree", "graph_pagerank", "graph_community", "graph_component_size"]
    return df[cols].astype("float32")


# ---------------------------------------------------------
# 9. Final GPU vector builder
# ---------------------------------------------------------

def build_gpu_vector(df):
    """
    df is a cuDF DataFrame already on GPU.
    Returns a cuDF DataFrame of numerical features.
    """

    logger.info(f"Building GPU vector for {len(df)} rows...")

    out = cudf.DataFrame()

    out = out.join(gpu_transform_amount(df["amount"], df.get("user_avg_amount")))
    out = out.join(gpu_time_features(df["timestamp"], df.get("last_timestamp")))
    out = out.join(gpu_device_features(df["device_id"], df["user_devices"], df["device_users"]))
    out = out.join(gpu_ip_features(df["ip_asn"], df["ip_is_proxy"], df["ip_risk_score"]))
    out = out.join(gpu_behavioral_features(df))
    out = out.join(gpu_risk_features(df["news_risk"], df["market_vol"], df["mempool_pressure"]))

    if {"graph_degree", "graph_pagerank", "graph_community", "graph_component_size"}.issubset(df.columns):
        out = out.join(gpu_graph_features(df))

    logger.info(f"GPU vector built with {out.shape[1]} features.")

    return out
