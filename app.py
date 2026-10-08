"""Small desktop window: fill in an order, press Predict, see the estimated minutes."""
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib

from predict import predict_minutes

HERE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(HERE, "delivery_model.joblib"))

root = tk.Tk()
root.title("Delivery Time Predictor")
root.resizable(False, False)
frm = ttk.Frame(root, padding=16)
frm.grid()

ttk.Label(frm, text="Delivery Time Prediction", font=("Segoe UI", 15, "bold")
          ).grid(columnspan=2, pady=(0, 10))

fields = {}


def add_entry(row, label, default):
    ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
    v = tk.StringVar(value=default)
    ttk.Entry(frm, textvariable=v, width=18).grid(row=row, column=1, pady=3)
    fields[label] = v


def add_combo(row, label, values, default):
    ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
    v = tk.StringVar(value=default)
    ttk.Combobox(frm, textvariable=v, values=values, state="readonly", width=16
                 ).grid(row=row, column=1, pady=3)
    fields[label] = v


add_entry(1, "Distance (km)", "7.5")
add_combo(2, "Traffic", ["Low", "Medium", "High", "Jam"], "High")
add_combo(3, "Weather", ["Clear", "Cloudy", "Fog", "Rain", "Storm"], "Rain")
add_combo(4, "Vehicle", ["Bicycle", "Scooter", "Motorcycle", "Car"], "Motorcycle")
add_combo(5, "Area", ["Urban", "Metropolitan", "Semiurban"], "Metropolitan")
add_entry(6, "Order hour (8-23)", "19")
add_combo(7, "Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"], "Saturday")
add_entry(8, "Preparation time (min)", "15")
add_entry(9, "Extra stops (0-3)", "1")
add_combo(10, "Festival day", ["No", "Yes"], "No")

result = tk.StringVar(value="")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def on_predict():
    try:
        g = lambda k: fields[k].get()
        dist = float(g("Distance (km)"))
        hour = int(g("Order hour (8-23)"))
        prep = int(g("Preparation time (min)"))
        stops = int(g("Extra stops (0-3)"))
        if not 0.5 <= dist <= 25:
            raise ValueError("Distance must be between 0.5 and 25 km")
        if not 8 <= hour <= 23:
            raise ValueError("Order hour must be between 8 and 23")
        if not 0 <= stops <= 3:
            raise ValueError("Extra stops must be between 0 and 3")
        order = {"order_hour": hour, "day_of_week": DAYS.index(g("Day")),
                 "distance_km": dist, "traffic_level": g("Traffic"),
                 "weather": g("Weather"), "vehicle_type": g("Vehicle"),
                 "area_type": g("Area"), "courier_experience_yrs": 3.0,
                 "courier_rating": 4.3, "prep_time_min": prep,
                 "multiple_deliveries": stops,
                 "festival": 1 if g("Festival day") == "Yes" else 0}
        result.set(f"Estimated delivery time: {predict_minutes(order, bundle):.1f} minutes")
    except ValueError as e:
        messagebox.showerror("Check your input", str(e))


ttk.Button(frm, text="Predict", command=on_predict).grid(row=11, columnspan=2, pady=10)
ttk.Label(frm, textvariable=result, font=("Segoe UI", 12, "bold"),
          foreground="#1f5fbf").grid(row=12, columnspan=2)

try:
    r = json.load(open(os.path.join(HERE, "results.json")))["final_test"]
    ttk.Label(frm, text=f"Model: {bundle['name']}   |   test MAE {r['MAE']:.2f} min, "
                        f"R² {r['R2']:.3f}", foreground="#666").grid(row=13, columnspan=2, pady=(8, 0))
except Exception:
    pass

root.mainloop()
