from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from risk_calculation.new_risk_eq_v2 import get_sports_heat_stress_curves

t = 36
rh = 40
r = risk_value = get_sports_heat_stress_curves(
        tdb=t, rh=rh, tr=t + 10, sport_id="running", v=1.0
    )
print(r)

results = []
for t, rh in product(np.arange(25, 45, 1), range(0, 101, 2)):
    risk_value = get_sports_heat_stress_curves(
        tdb=t, rh=rh, tr=t + 10, sport_id="running", v=1.0
    )
    results.append((t, rh, risk_value))

df = pd.DataFrame(results, columns=["tdb", "rh", "risk"])

# Pivot the data to create a matrix suitable for heatmap
heatmap_data = df.pivot(index="rh", columns="tdb", values="risk")
heatmap_data.sort_index(inplace=True, ascending=False)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap="RdYlGn_r",
    cbar_kws={"label": "Risk Level"},
    xticklabels=True,
    yticklabels=True,
)
plt.xlabel("Temperature (°C)")
plt.ylabel("Relative Humidity (%)")
plt.title("Heat Stress Risk Levels by Temperature and Humidity (Running)")
plt.tight_layout()
plt.savefig("figures/heatmap_risk.png", dpi=300, bbox_inches="tight")
plt.show()