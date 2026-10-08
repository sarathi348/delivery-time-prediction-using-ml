"""Generate the figures used in the report from the saved experiment outputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a76d2", "#ec6a36"
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})

df = pd.read_csv("delivery_clean.csv")
pred = pd.read_csv("test_predictions.csv")
res = json.load(open("results.json"))
T = "delivery_time_min"

# Fig 5.1
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].hist(df[T], bins=45, color=BLUE, edgecolor="white")
ax[0].axvline(df[T].median(), color="k", ls="--")
ax[0].text(df[T].median() + 2, ax[0].get_ylim()[1] * 0.93,
           f"median {df[T].median():.1f} min", fontsize=8)
ax[0].set(title="(a) Distribution of the target", xlabel="Delivery time (minutes)",
          ylabel="Number of orders")
s = df.sample(1500, random_state=1)
ax[1].scatter(s["distance_km"], s[T], s=6, alpha=0.4, color=BLUE)
ax[1].set(title="(b) Distance against delivery time", xlabel="Distance (km)",
          ylabel="Delivery time (minutes)")
plt.tight_layout(); plt.savefig("fig5_1_distribution.png", dpi=200); plt.close()

# Fig 5.2
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, col, order, title in [
        (ax[0], "traffic_level", ["Low", "Medium", "High", "Jam"], "(a) By traffic level"),
        (ax[1], "weather", ["Clear", "Cloudy", "Fog", "Rain", "Storm"], "(b) By weather")]:
    m = df.groupby(col)[T].mean().reindex(order)
    bars = a.bar(order, m.values, color=BLUE, width=0.55)
    a.bar_label(bars, fmt="%.1f", fontsize=8)
    a.set(title=title, ylabel="Mean delivery time (minutes)")
plt.tight_layout(); plt.savefig("fig5_2_traffic_weather.png", dpi=200); plt.close()

# Fig 5.3
imp = pd.Series(res["permutation_importance"]).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.barh(imp.index, imp.values, color=BLUE)
ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xlabel("Increase in test RMSE when the feature is shuffled (minutes)")
plt.tight_layout(); plt.savefig("fig5_3_importance.png", dpi=200); plt.close()

# Fig 6.1
names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
it2 = res["iteration2"]
mae = [it2[n]["test"]["MAE"] for n in names]
rmse = [it2[n]["test"]["RMSE"] for n in names]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = range(len(names))
b1 = ax.bar([i - 0.2 for i in x], mae, 0.38, label="MAE", color=BLUE)
b2 = ax.bar([i + 0.2 for i in x], rmse, 0.38, label="RMSE", color=ORANGE)
ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("Error on the test set (minutes)"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("fig6_1_model_comparison.png", dpi=200); plt.close()

# Fig 6.2
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(pred["actual"], pred["predicted"], s=6, alpha=0.4, color=BLUE)
lim = [pred.min().min() - 2, pred.max().max() + 2]
ax.plot(lim, lim, "k--", lw=1)
ax.set(xlabel="Actual delivery time (minutes)", ylabel="Predicted delivery time (minutes)",
       xlim=lim, ylim=lim)
plt.tight_layout(); plt.savefig("fig6_2_actual_vs_predicted.png", dpi=200); plt.close()
print("Saved 5 figures")
