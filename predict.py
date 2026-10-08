"""Command-line interface: estimate delivery time for one new order."""
import argparse

import joblib
import pandas as pd

from features import engineer


def predict_minutes(order, bundle):
    X = engineer(pd.DataFrame([order]), bundle["fill"])
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    return float(bundle["model"].predict(X)[0])


def main():
    p = argparse.ArgumentParser(description="Delivery time estimator")
    p.add_argument("--distance", type=float, required=True, help="distance in km (0.5-25)")
    p.add_argument("--traffic", choices=["Low", "Medium", "High", "Jam"], required=True)
    p.add_argument("--weather", choices=["Clear", "Cloudy", "Fog", "Rain", "Storm"], default="Clear")
    p.add_argument("--vehicle", choices=["Bicycle", "Scooter", "Motorcycle", "Car"], default="Motorcycle")
    p.add_argument("--area", choices=["Urban", "Metropolitan", "Semiurban"], default="Urban")
    p.add_argument("--hour", type=int, default=12, help="order hour (8-23)")
    p.add_argument("--day", type=int, default=0, help="0=Monday ... 6=Sunday")
    p.add_argument("--prep", type=int, default=15, help="preparation time in minutes")
    p.add_argument("--stops", type=int, default=0, help="extra deliveries on the trip (0-3)")
    p.add_argument("--festival", type=int, choices=[0, 1], default=0)
    p.add_argument("--experience", type=float, default=3.0)
    p.add_argument("--rating", type=float, default=4.3)
    p.add_argument("--model", default="delivery_model.joblib")
    a = p.parse_args()

    if not 0.5 <= a.distance <= 25:
        p.error("distance must be between 0.5 and 25 km")

    bundle = joblib.load(a.model)
    order = {"order_hour": a.hour, "day_of_week": a.day, "distance_km": a.distance,
             "traffic_level": a.traffic, "weather": a.weather, "vehicle_type": a.vehicle,
             "area_type": a.area, "courier_experience_yrs": a.experience,
             "courier_rating": a.rating, "prep_time_min": a.prep,
             "multiple_deliveries": a.stops, "festival": a.festival}
    print(f"Estimated delivery time: {predict_minutes(order, bundle):.1f} minutes")


if __name__ == "__main__":
    main()
