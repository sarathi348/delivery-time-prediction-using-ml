"""Feature engineering shared by training and the prediction interface."""
import numpy as np
import pandas as pd

TRAFFIC = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
WEATHER = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 2, "Storm": 3}
RUSH_HOURS = [8, 9, 10, 17, 18, 19, 20]
NOMINAL = ["traffic_level", "weather", "vehicle_type", "area_type"]
NUMERIC = ["order_hour", "day_of_week", "distance_km", "courier_experience_yrs",
           "courier_rating", "prep_time_min", "multiple_deliveries", "festival"]
FEATURE_SRC = NUMERIC + NOMINAL


def basic(d, fill):
    """Iteration 1: fill missing values and one-hot encode (24 inputs)."""
    d = d[FEATURE_SRC].fillna(fill).copy()
    return pd.get_dummies(d, columns=NOMINAL, dtype=float)


def engineer(d, fill):
    """Iteration 2: add ten engineered features (34 inputs)."""
    d = d[FEATURE_SRC].fillna(fill).copy()
    d["traffic_index"] = d["traffic_level"].map(TRAFFIC)
    d["weather_severity"] = d["weather"].map(WEATHER)
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["is_rush_hour"] = d["order_hour"].isin(RUSH_HOURS).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["order_hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["order_hour"] / 24)
    d["dist_x_traffic"] = d["distance_km"] * d["traffic_index"]
    d["dist_x_weather"] = d["distance_km"] * (1 + d["weather_severity"])
    d["dist_x_bicycle"] = d["distance_km"] * (d["vehicle_type"] == "Bicycle")
    d["stops_x_traffic"] = d["multiple_deliveries"] * d["traffic_index"]
    return pd.get_dummies(d, columns=NOMINAL, dtype=float)
