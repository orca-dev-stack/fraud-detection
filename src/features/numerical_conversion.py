import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("numerical_conversion")


# ---------------------------------------------------------
# 1. Amount transformations
# ---------------------------------------------------------

def transform_amount(amount: float, user_avg: float = None):
    """Return log amount, z-score, and ratio to user average."""
    log_amt = np.log1p(amount)

    if user_avg and user_avg > 0:
        ratio = amount / user_avg
    else:
        ratio = 1.0

    return {
        "log_amount": log_amt,
        "amount_ratio": ratio,
    }


# ---------------------------------------------------------
# 2. Time-based features
# ---------------------------------------------------------

def time_features(timestamp: str, last_timestamp: str = None):
    """Convert timestamps into behavioral time features."""
    ts = pd.to_datetime(timestamp)

    hour = ts.hour
    day_of_week = ts.weekday()

    if last_timestamp:
        last_ts = pd.to_datetime(last_timestamp)
        delta = (ts - last_ts).total_seconds()
    else:
        delta = 999999  # first transaction

    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "time_since_last_tx": delta,
    }


# ---------------------------------------------------------
# 3. Velocity features
# ---------------------------------------------------------

def velocity_features(tx_times: list):
    """
    Compute transaction velocity:
    - per minute
    - per hour
    - per day
    """
    if len(tx_times) < 2:
        return {
            "tx_per_min": 0,
            "tx_per_hour": 0,
            "tx_per_day": 0,
        }

    ts = pd.to_datetime(tx_times)
    diffs = (ts.max() - ts.min()).total_seconds()

    if diffs == 0:
        return {
            "tx_per_min": len(ts),
            "tx_per_hour": len(ts),
            "tx_per_day": len(ts),
        }

    return {
        "tx_per_min": len(ts) / (diffs / 60),
        "tx_per_hour": len(ts) / (diffs / 3600),
        "tx_per_day": len(ts) / (diffs / 86400),
    }


# ---------------------------------------------------------
# 4. Device features
# ---------------------------------------------------------

def device_features(device_id: str, user_devices: list, device_users: list):
    """
    Convert device info into numerical signals.
    """
    return {
        "device_seen_before": 1 if device_id in user_devices else 0,
        "device_user_count": len(device_users),
        "user_device_count": len(user_devices),
    }


# ---------------------------------------------------------
# 5. IP features
# ---------------------------------------------------------

def ip_features(ip_info: dict):
    """
    ip_info example:
    {
        "asn": 12345,
        "country": "US",
        "is_proxy": True,
        "risk_score": 0.82
    }
    """
    return {
        "ip_asn": ip_info.get("asn", 0),
        "ip_is_proxy": int(ip_info.get("is_proxy", False)),
        "ip_risk_score": float(ip_info.get("risk_score", 0.0)),
    }


# ---------------------------------------------------------
# 6. Behavioral features
# ---------------------------------------------------------

def behavioral_features(history: dict):
    """
    history example:
    {
        "avg_amount": 42.1,
        "std_amount": 10.3,
        "merchant_diversity": 12,
        "country_diversity": 3,
        "device_switch_rate": 0.18
    }
    """
    return {
        "avg_amount": history.get("avg_amount", 0),
        "std_amount": history.get("std_amount", 0),
        "merchant_diversity": history.get("merchant_diversity", 0),
        "country_diversity": history.get("country_diversity", 0),
        "device_switch_rate": history.get("device_switch_rate", 0),
    }


# ---------------------------------------------------------
# 7. Risk features (news, market, mempool)
# ---------------------------------------------------------

def risk_features(news_risk, market_vol, mempool_pressure):
    return {
        "news_risk": float(news_risk),
        "market_volatility": float(market_vol),
        "mempool_pressure": float(mempool_pressure),
    }


# ---------------------------------------------------------
# 8. Graph features (optional)
# ---------------------------------------------------------

def graph_features(graph_info: dict):
    """
    graph_info example:
    {
        "degree": 4,
        "pagerank": 0.0021,
        "community": 12,
        "component_size": 44
    }
    """
    return {
        "graph_degree": graph_info.get("degree", 0),
        "graph_pagerank": graph_info.get("pagerank", 0),
        "graph_community": graph_info.get("community", 0),
        "graph_component_size": graph_info.get("component_size", 1),
    }


# ---------------------------------------------------------
# 9. Final vector builder
# ---------------------------------------------------------

def build_numeric_vector(
    amount,
    timestamp,
    last_timestamp,
    tx_times,
    device_id,
    user_devices,
    device_users,
    ip_info,
    behavior,
    news_risk,
    market_vol,
    mempool_pressure,
    graph_info=None,
    user_avg_amount=None,
):
    """
    Build a full numerical feature vector for a transaction.
    """

    features = {}

    features.update(transform_amount(amount, user_avg_amount))
    features.update(time_features(timestamp, last_timestamp))
    features.update(velocity_features(tx_times))
    features.update(device_features(device_id, user_devices, device_users))
    features.update(ip_features(ip_info))
    features.update(behavioral_features(behavior))
    features.update(risk_features(news_risk, market_vol, mempool_pressure))

    if graph_info:
        features.update(graph_features(graph_info))

    logger.info(f"Numeric vector built with {len(features)} features.")
    return features
