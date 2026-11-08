"""Synthetic data generator — Day 17: causal impact of a staggered store remodel.

A fictional Azerbaijani fuel-station network runs a convenience shop at each of
60 stations. 24 shops were remodelled to a new format in three waves
(Apr 2025, Oct 2025, Apr 2026); 36 were never remodelled. Each shop closes for
two weeks of construction before reopening.

Outputs (fixed seed):
  stations.csv      station master + remodel log
  weekly_sales.csv  station x week revenue / transactions (with realistic dirt)
  ground_truth.csv  true log-effect of the remodel per station-week (validation only)
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(17)
OUT = Path(__file__).resolve().parent

weeks = pd.date_range("2024-01-01", "2026-10-26", freq="W-MON")
T = len(weeks)
t_years = np.arange(T) / 52.18

# ---------------------------------------------------------------- stations
types = ["Baku city"] * 30 + ["Highway"] * 15 + ["Regional town"] * 15
regions_by_type = {
    "Baku city": ["Nasimi", "Yasamal", "Narimanov", "Khatai", "Sabunchu", "Binagadi", "Khazar", "Surakhani"],
    "Highway": ["Baku-Quba M1", "Baku-Shamakhi M4", "Baku-Alat M3", "Ganja-Gazakh M2", "Baku-Ganja M2"],
    "Regional town": ["Ganja", "Sumgait", "Mingachevir", "Shaki", "Lankaran", "Shirvan", "Quba", "Gabala"],
}
n = len(types)
st = pd.DataFrame({
    "station_id": [f"ST-{i:02d}" for i in range(1, n + 1)],
    "location_type": types,
})
st["area"] = [rng.choice(regions_by_type[t]) for t in st.location_type]
base_level = {"Baku city": 18_000, "Highway": 26_000, "Regional town": 12_000}
st["log_base"] = np.log(st.location_type.map(base_level)) + rng.normal(0, 0.30, n)
st["shop_m2"] = np.round(np.exp(st.log_base - np.log(st.location_type.map(base_level))) *
                         st.location_type.map({"Baku city": 120, "Highway": 160, "Regional town": 100}) + rng.normal(0, 8, n))

# remodel assignment: management picks bigger shops first (selection on level)
waves = {"2025-04-07": {"Highway": 4, "Baku city": 2, "Regional town": 2},
         "2025-10-06": {"Highway": 3, "Baku city": 3, "Regional town": 2},
         "2026-04-06": {"Highway": 3, "Baku city": 3, "Regional town": 2}}
st["remodel_date"] = pd.NaT
for d, quota in waves.items():
    for typ, k in quota.items():
        pool = st[(st.location_type == typ) & st.remodel_date.isna()]
        p = np.exp(2.5 * (pool.log_base - pool.log_base.mean()))
        pick = rng.choice(pool.index, size=k, replace=False, p=p / p.sum())
        st.loc[pick, "remodel_date"] = pd.Timestamp(d)
capex_mu = {"Baku city": 185_000, "Highway": 230_000, "Regional town": 150_000}
st["remodel_capex_azn"] = np.where(st.remodel_date.notna(),
                                   np.round(st.location_type.map(capex_mu) * rng.uniform(0.9, 1.1, n), -3), np.nan)
# flagship: the largest wave-1 highway station got a drive-through as well
w1_hw = st[(st.remodel_date == "2025-04-07") & (st.location_type == "Highway")]
flagship = w1_hw.log_base.idxmax()
st["format"] = np.where(st.remodel_date.isna(), "old", "new")
st.loc[flagship, "format"] = "new + drive-through"
st.loc[flagship, "remodel_capex_azn"] += 90_000

# --------------------------------------------------------------- outcomes
woy = weeks.isocalendar().week.to_numpy().astype(float)
phase = 2 * np.pi * (woy - 29) / 52  # peak late July
season = {"Baku city": -0.05 * np.cos(phase), "Highway": 0.20 * np.cos(phase),
          "Regional town": 0.08 * np.cos(phase)}
common = 0.045 * t_years
common += np.where(woy >= 52, 0.22, 0) + np.where(woy == 1, 0.10, 0)       # New Year
common += np.where(woy == 12, 0.16, 0)                                      # Novruz
common += np.where(weeks >= "2026-01-12", -0.05, 0)                         # fuel price hike

eff_type = {"Baku city": 0.05, "Highway": 0.15, "Regional town": 0.10}
rows, truth = [], []
for i, s in st.iterrows():
    slope = rng.normal(0, 0.015)
    eps = np.zeros(T)
    e = rng.normal(0, 0.055, T)
    for k in range(T):
        eps[k] = (0.5 * eps[k - 1] if k else 0) + e[k]
    y = s.log_base + common + season[s.location_type] + slope * t_years + eps
    tau = np.zeros(T)
    closed = np.zeros(T, bool)
    if pd.notna(s.remodel_date):
        g = weeks.get_loc(s.remodel_date)
        k_since = np.arange(T) - g
        full = eff_type[s.location_type] * np.exp(rng.normal(0, 0.25))
        if i == flagship:
            full = 0.28
        tau = np.where(k_since >= 0, full * (1 - np.exp(-(k_since + 1) / 6)), 0.0)
        closed = (k_since == -1) | (k_since == -2)
    y = y + tau
    rev = np.exp(y)
    basket0 = {"Baku city": 11.5, "Highway": 14.0, "Regional town": 9.5}[s.location_type]
    basket = basket0 * np.exp(0.4 * tau + rng.normal(0, 0.02, T))
    trx = np.round(rev / basket)
    rev = np.round(trx * basket, 2)
    trx[closed] = 0
    rev[closed] = 0.0
    rows.append(pd.DataFrame({"station_id": s.station_id, "week_start": weeks,
                              "revenue_azn": rev, "transactions": trx.astype(int)}))
    truth.append(pd.DataFrame({"station_id": s.station_id, "week_start": weeks,
                               "true_log_effect": tau, "closed_for_remodel": closed}))

sales = pd.concat(rows, ignore_index=True)
gt = pd.concat(truth, ignore_index=True)

# ----------------------------------------------------------------- dirt
drop = rng.random(len(sales)) < 0.015                          # feed gaps
drop &= ~gt.closed_for_remodel.to_numpy()
sales = sales[~drop].copy()
qepik = rng.random(len(sales)) < 0.004                         # revenue sent in qəpik
sales.loc[qepik & (sales.revenue_azn > 0), "revenue_azn"] *= 100
idx = rng.random(len(sales)) < 0.04                            # messy station codes
fmt = rng.integers(0, 3, len(sales))
ids = sales.station_id.to_numpy().copy()
for j in np.where(idx)[0]:
    num = ids[j][3:]
    ids[j] = [f"st-{num}", f"ST{num}", f" ST-{num} "][fmt[j]]
sales["station_id"] = ids
dups = sales.sample(180, random_state=3)                       # double upload
sales = pd.concat([sales, dups]).sample(frac=1, random_state=4).reset_index(drop=True)
sales["week_start"] = sales.week_start.dt.strftime("%Y-%m-%d")

st_out = st.drop(columns=["log_base"]).copy()
st_out["remodel_date"] = st_out.remodel_date.dt.strftime("%Y-%m-%d")
st_out.to_csv(OUT / "stations.csv", index=False)
sales.to_csv(OUT / "weekly_sales.csv", index=False)
gt.to_csv(OUT / "ground_truth.csv", index=False)
print(st_out.groupby(["location_type", "remodel_date"], dropna=False).size())
print(sales.shape, gt.shape, "flagship:", st.loc[flagship, "station_id"])
