"""Cleaning + causal-inference helpers for Day 17 (written from scratch with NumPy/SciPy)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA = Path(__file__).resolve().parent / "data"
PRE_WIN = (-10, -3)     # baseline window, weeks relative to go-live (construction = -2, -1)
TYPES = ["Baku city", "Highway", "Regional town"]


# ------------------------------------------------------------------ loading & cleaning
def load_raw():
    return (pd.read_csv(DATA / "weekly_sales.csv"),
            pd.read_csv(DATA / "stations.csv", parse_dates=["remodel_date"]))


def clean(sales_raw, st):
    s = sales_raw.copy()
    log = {"raw rows": len(s)}
    norm = s.station_id.str.strip().str.upper().str.replace(r"^ST-?", "ST-", regex=True)
    log["station codes normalised"] = int((norm != s.station_id).sum())
    s["station_id"] = norm
    s["week_start"] = pd.to_datetime(s.week_start)
    before = len(s)
    s = s.drop_duplicates(["station_id", "week_start"])
    log["duplicate rows dropped"] = before - len(s)
    # revenue sent in qəpik: implied basket ~100x the station's median basket
    basket = s.revenue_azn / s.transactions.replace(0, np.nan)
    med = basket.groupby(s.station_id).transform("median")
    bad = basket > 20 * med
    log["revenue rows in qəpik (÷100)"] = int(bad.sum())
    s.loc[bad, "revenue_azn"] /= 100
    # construction weeks from the remodel log
    s = s.merge(st[["station_id", "location_type", "remodel_date"]], on="station_id", how="left")
    s["rel_week"] = ((s.week_start - s.remodel_date).dt.days // 7)
    s["closed"] = s.rel_week.isin([-2, -1])
    zero = s.revenue_azn <= 0
    log["zero-revenue weeks"] = int(zero.sum())
    log["…of which match the remodel log (construction)"] = int((zero & s.closed).sum())
    weeks = pd.date_range(s.week_start.min(), s.week_start.max(), freq="W-MON")
    log["missing station-weeks (feed gaps)"] = len(st) * len(weeks) - len(s)
    s["log_rev"] = np.where(s.closed | zero, np.nan, np.log(s.revenue_azn.where(~zero)))
    s["treated"] = s.remodel_date.notna()
    s["post"] = s.rel_week >= 0
    return s.sort_values(["station_id", "week_start"]).reset_index(drop=True), log


def wide(panel, col="log_rev"):
    return panel.pivot(index="station_id", columns="week_start", values=col)


# ------------------------------------------------------------------ TWFE with cluster-robust SE
def ols_cluster(y, X, groups):
    """OLS with Liang-Zeger cluster-robust (CR1) standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        sg = X[m].T @ u[m]
        meat += np.outer(sg, sg)
    G, N, K = len(np.unique(groups)), len(y), X.shape[1]
    c = G / (G - 1) * (N - 1) / (N - K)
    V = c * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def twfe(panel, extra=None):
    d = panel.dropna(subset=["log_rev"]).copy()
    d["D"] = (d.treated & d.post).astype(float)
    S = pd.get_dummies(d.station_id, drop_first=True, dtype=float)
    if extra == "type_week":
        W = pd.get_dummies(d.location_type + "|" + d.week_start.astype(str), drop_first=True, dtype=float)
    else:
        W = pd.get_dummies(d.week_start, drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d.D, S, W])
    b, se = ols_cluster(d.log_rev.to_numpy(), X, d.station_id.to_numpy())
    return b[1], se[1]


# ------------------------------------------------------------------ DiD imputation (Callaway-Sant'Anna style)
def unit_effects(Y, st, stratify=True):
    """Per treated station-week effect: (y_it - baseline_i) - mean over never-treated
    controls [same location type if stratify] of (y_jt - baseline_j), baseline = mean of
    weeks PRE_WIN relative to the cohort's go-live. Returns DataFrame stations x weeks."""
    weeks = Y.columns
    st = st.set_index("station_id").loc[Y.index]
    never = st.remodel_date.isna().to_numpy()
    out = pd.DataFrame(np.nan, index=Y.index, columns=weeks)
    Yv = Y.to_numpy()
    for g in st.remodel_date.dropna().unique():
        gi = weeks.get_loc(g)
        base = np.nanmean(Yv[:, gi + PRE_WIN[0]: gi + PRE_WIN[1] + 1], axis=1, keepdims=True)
        dev = Yv - base
        coh = (st.remodel_date == g).to_numpy()
        for typ in (TYPES if stratify else [None]):
            m_t = coh & ((st.location_type == typ).to_numpy() if typ else True)
            m_c = never & ((st.location_type == typ).to_numpy() if typ else True)
            if m_t.sum() == 0:
                continue
            ctrl = np.nanmean(dev[m_c], axis=0)
            out.iloc[np.where(m_t)[0]] = dev[m_t] - ctrl
    return out


def rel_week_matrix(Y, st):
    st = st.set_index("station_id").loc[Y.index]
    w = np.asarray(Y.columns.values, dtype="datetime64[D]")
    g = st.remodel_date.to_numpy(dtype="datetime64[D]")
    rel = (w[None, :] - g[:, None]).astype("timedelta64[D]").astype(float) / 7
    return pd.DataFrame(rel, index=Y.index, columns=Y.columns)


def event_study(tau, rel, emin=-30, emax=60):
    t = tau.to_numpy().ravel(); r = rel.to_numpy().ravel()
    ok = ~np.isnan(t) & ~np.isnan(r) & (r >= emin) & (r <= emax)
    return pd.Series(t[ok]).groupby(r[ok].astype(int)).mean()


def overall_att(tau, rel):
    t = tau.to_numpy(); r = rel.to_numpy()
    return np.nanmean(np.where(r >= 0, t, np.nan))


def cluster_bootstrap(Y, st, stat, B=300, seed=0):
    """Resample stations with replacement within (treated cohort / never-treated) x type cells."""
    rng = np.random.default_rng(seed)
    st = st.set_index("station_id").loc[Y.index].reset_index()
    st["cell"] = st.remodel_date.dt.strftime("%Y-%m-%d").fillna("never") + "|" + st.location_type
    cells = [np.where(st.cell == c)[0] for c in st.cell.unique()]
    res = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(c, len(c)) for c in cells])
        Yb = Y.iloc[idx].copy(); stb = st.iloc[idx].copy()
        new_ids = [f"b{k}" for k in range(len(idx))]
        Yb.index = new_ids; stb["station_id"] = new_ids
        res.append(stat(Yb, stb))
    return np.array(res)


# ------------------------------------------------------------------ synthetic control
def synth_weights(y1_pre, Y0_pre):
    """Demeaned synthetic control: convex weights on donors matching the treated unit's
    pre-period deviations from its own mean (allows a level difference)."""
    a = y1_pre - y1_pre.mean()
    B = Y0_pre - Y0_pre.mean(axis=1, keepdims=True)
    J = B.shape[0]
    f = lambda w: np.sum((a - w @ B) ** 2)
    jac = lambda w: -2 * B @ (a - w @ B)
    r = minimize(f, np.full(J, 1 / J), jac=jac, method="SLSQP", bounds=[(0, 1)] * J,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                 options={"maxiter": 500, "ftol": 1e-12})
    return r.x


def synth_path(Y, treated_id, donors, pre_mask):
    y1 = Y.loc[treated_id].to_numpy(); Y0 = Y.loc[donors].to_numpy()
    ok = pre_mask & ~np.isnan(y1) & ~np.isnan(Y0).any(axis=0)
    w = synth_weights(y1[ok], Y0[:, ok])
    # fill donor gaps with the donor's own interpolation for the path
    Y0f = pd.DataFrame(Y0).T.interpolate(limit_direction="both").T.to_numpy()
    synth = w @ (Y0f - Y0f[:, ok].mean(axis=1, keepdims=True)) + y1[ok].mean()
    return w, synth


def rmspe(gap, mask):
    g = gap[mask]; g = g[~np.isnan(g)]
    return np.sqrt(np.mean(g ** 2))
