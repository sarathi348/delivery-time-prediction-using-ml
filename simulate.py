"""Rule-based generator for the delivery dataset (12,120 rows, 14 columns).

Travel speed depends on vehicle, traffic, weather and area. Random noise,
missing values, duplicate rows and a few faulty records are added on purpose
so that the cleaning stage has real work to do.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 12000

SPEED = {"Bicycle": 14, "Scooter": 22, "Motorcycle": 28, "Car": 24}   # km/h
TRAFFIC_F = {"Low": 1.0, "Medium": 0.88, "High": 0.76, "Jam": 0.60}
WEATHER_F = {"Clear": 1.0, "Cloudy": 0.98, "Fog": 0.89, "Rain": 0.88, "Storm": 0.78}
AREA_F = {"Urban": 1.0, "Metropolitan": 0.90, "Semiurban": 1.08}


def generate(seed=SEED, n=N):
    rng = np.random.default_rng(seed)
    hour = rng.integers(8, 24, n)
    day = rng.integers(0, 7, n)
    dist = np.clip(rng.gamma(2.5, 2.3, n), 0.5, 25).round(2)

    rush = np.isin(hour, [8, 9, 10, 17, 18, 19, 20])
    traffic = np.where(
        rush,
        rng.choice(list(TRAFFIC_F), n, p=[0.10, 0.25, 0.35, 0.30]),
        rng.choice(list(TRAFFIC_F), n, p=[0.35, 0.35, 0.20, 0.10]),
    )
    weather = rng.choice(list(WEATHER_F), n, p=[0.35, 0.25, 0.12, 0.18, 0.10])
    vehicle = rng.choice(list(SPEED), n, p=[0.10, 0.30, 0.45, 0.15])
    area = rng.choice(list(AREA_F), n, p=[0.45, 0.35, 0.20])
    exp = rng.integers(0, 13, n)
    rating = np.clip(rng.normal(4.3, 0.4, n), 3.0, 5.0).round(1)
    prep = np.clip(rng.normal(15, 7, n), 4, 35).round().astype(int)
    multi = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.30, 0.18, 0.07])
    festival = (rng.random(n) < 0.08).astype(int)

    speed = (np.array([SPEED[v] for v in vehicle])
             * np.array([TRAFFIC_F[t] for t in traffic])
             * np.array([WEATHER_F[w] for w in weather])
             * np.array([AREA_F[a] for a in area]))
    travel = dist / speed * 60
    time = (travel + 1.0 * prep + 5.0 * multi + 3.0 * festival
            + rng.normal(0, 0.06 * travel + 2.0, n))
    time = np.clip(time, 8, None).round(1)

    df = pd.DataFrame({
        "order_id": [f"ORD{i:05d}" for i in range(n)],
        "order_hour": hour, "day_of_week": day, "distance_km": dist,
        "traffic_level": traffic, "weather": weather,
        "vehicle_type": vehicle, "area_type": area,
        "courier_experience_yrs": exp, "courier_rating": rating,
        "prep_time_min": prep, "multiple_deliveries": multi,
        "festival": festival, "delivery_time_min": time,
    })

    # Deliberate faults ------------------------------------------------
    bad = rng.choice(n, 20, replace=False)
    df.loc[bad, "distance_km"] = -df.loc[bad, "distance_km"]          # impossible
    for col, k in [("courier_rating", 237), ("weather", 177), ("traffic_level", 120)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan         # missing
    dup = df.sample(120, random_state=seed)                           # duplicates
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    data = generate()
    data.to_csv("delivery_data.csv", index=False)
    print(f"Wrote delivery_data.csv with {data.shape[0]} rows, {data.shape[1]} columns")
