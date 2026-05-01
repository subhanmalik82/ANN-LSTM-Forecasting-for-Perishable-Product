# streamlit_forecast.py
# Visualize and forecast time-series data (ANN, LSTM, ARIMA) with Streamlit
# Includes: Resampling, Hampel filtering, EWMA smoothing, hyperparameter tuning,
# and 4-panel forecasting for CO2, CH4, NH3, H2S.

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ARIMA (statsmodels)
SM_AVAILABLE = True
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    SM_AVAILABLE = False

# TensorFlow for LSTM (optional)
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Time-series Forecast (ANN/LSTM/ARIMA)", layout="wide")

# -------------------------------
# Utilities
# -------------------------------
def fingerprint_series(series: pd.Series, extra: dict) -> str:
    """Fingerprint to invalidate cache when data/settings change."""
    h = hashlib.sha1()
    hv = pd.util.hash_pandas_object(series, index=True).values
    h.update(hv.tobytes())
    h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def tune_ann_gridsearch(
    fingerprint: str,
    Xtr_flat: np.ndarray,
    ytr_s: np.ndarray,
    param_grid: dict,
    n_splits: int = 3,
    random_state: int = 42,
):
    """
    TimeSeriesSplit GridSearch on TRAIN only (scaled).
    Returns best_params, best_estimator, best_score (neg MSE).
    """
    n_samples = len(Xtr_flat)
    if n_samples < 10:
        raise ValueError("Not enough training samples for ANN GridSearchCV.")

    n_splits_eff = min(n_splits, max(2, n_samples // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits_eff)

    base = MLPRegressor(
        max_iter=600,
        random_state=random_state,
        early_stopping=False,  # keep CV behavior consistent
    )

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(Xtr_flat, ytr_s.ravel())
    return gs.best_params_, gs.best_estimator_, float(gs.best_score_)


@st.cache_resource(show_spinner=False)
def tune_lstm_cached(
    fp: str,
    Xtr_l_s: np.ndarray,
    ytr_s: np.ndarray,
    Xva_l_s: np.ndarray,
    yva_s: np.ndarray,
    grid: list,
    max_epochs: int = 60,
    patience: int = 8,
):
    """
    Manual grid search for LSTM (best by val_loss if validation exists).
    Returns best_params + best_val_loss.
    """
    if not TF_AVAILABLE:
        return None, np.nan

    has_val = (Xva_l_s is not None) and (len(Xva_l_s) > 0)
    best_params, best_val = None, np.inf

    for params in grid:
        tf.keras.backend.clear_session()

        units = int(params["units"])
        dense_units = int(params["dense_units"])
        lr = float(params["lr"])
        batch_size = int(params["batch_size"])

        model = Sequential([
            LSTM(units, input_shape=(Xtr_l_s.shape[1], Xtr_l_s.shape[2])),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

        monitor_metric = "val_loss" if has_val else "loss"
        es = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
        )

        hist = model.fit(
            Xtr_l_s, ytr_s,
            validation_data=(Xva_l_s, yva_s) if has_val else None,
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )

        score = float(np.min(hist.history["val_loss"])) if has_val else float(np.min(hist.history["loss"]))
        if score < best_val:
            best_val = score
            best_params = params

    return best_params, float(best_val)


def find_datetime_col(df: pd.DataFrame):
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(key in cl for key in ["time", "date", "timestamp", "tgl"]):
            candidates.append(c)

    if not candidates:
        for c in df.columns:
            try:
                if np.issubdtype(df[c].dtype, np.datetime64):
                    candidates.append(c)
            except Exception:
                pass

    return candidates[0] if candidates else None


def to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            coerced = pd.to_numeric(out[c], errors="coerce")
            if coerced.notna().sum() >= max(5, int(0.1 * len(out))):
                out[c] = coerced
    return out


def rolling_mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return 1.4826 * np.median(np.abs(arr - med))


def hampel_outliers(s: pd.Series, window: int = 9, n_sigmas: float = 3.0) -> pd.Series:
    if s.isna().all():
        return pd.Series(False, index=s.index)
    med = s.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    mad = s.rolling(window=window, center=True, min_periods=max(3, window // 2)).apply(rolling_mad, raw=True)
    diff = (s - med).abs()
    thresh = n_sigmas * mad.replace(0, np.nan)
    return (diff > thresh).fillna(False)


def compute_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_pred = np.asarray(y_pred).reshape(-1).astype(float)

    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE_%": np.nan}

    err = y_true - y_pred

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else np.nan

    denom = np.abs(y_true)
    mask = denom > eps
    mape = float(np.mean(np.abs(err[mask]) / denom[mask]) * 100) if np.any(mask) else np.nan

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def make_supervised(series: pd.Series, lookback: int, horizon: int):
    values = series.values.astype("float32")
    idx = series.index
    X, y, ts = [], [], []
    for t in range(lookback, len(values) - horizon + 1):
        X.append(values[t - lookback:t])
        y.append(values[t + horizon - 1])
        ts.append(idx[t + horizon - 1])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    ts = np.array(ts)
    X_lstm = X[..., None]  # (N, lookback, 1)
    return X, X_lstm, y, ts


def time_split_safe(X, X_lstm, y, ts, val_frac: float, test_frac: float, allow_zero_val=True):
    n = len(y)
    if n <= 1:
        raise ValueError(f"Not enough supervised samples (N={n}). Reduce lookback/horizon.")

    vf = max(0.0, min(0.4, float(val_frac)))
    tf = max(0.05, min(0.5, float(test_frac)))

    while True:
        n_test = int(np.floor(tf * n))
        n_val = int(np.floor(vf * n))
        n_train = n - n_val - n_test

        if n_test == 0 and tf > 0:
            n_test = 1
            n_train = n - n_val - n_test

        if n_train > 0:
            break

        if vf > 0.0 and allow_zero_val:
            vf = max(0.0, vf - 0.05)
        elif tf > 0.05:
            tf = max(0.05, tf - 0.05)
        else:
            raise ValueError(f"Not enough data for splits after shrinking (n={n}, vf={vf:.2f}, tf={tf:.2f}).")

    split = {
        "train": (X[:n_train], X_lstm[:n_train], y[:n_train], ts[:n_train]),
        "val": (X[n_train:n_train + n_val], X_lstm[n_train:n_train + n_val], y[n_train:n_train + n_val], ts[n_train:n_train + n_val]),
        "test": (X[-n_test:], X_lstm[-n_test:], y[-n_test:], ts[-n_test:]),
    }
    info = {"N": n, "train": n_train, "val": n_val, "test": n_test, "vf": vf, "tf": tf}
    return split, info


def plot_forecast(ts, y_true, y_pred, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=np.asarray(y_true).ravel(), mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=ts, y=np.asarray(y_pred).ravel(), mode="lines", name="Forecast"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_future_single(history_s, future_s, title):
    fig = go.Figure()
    if history_s is not None and len(history_s):
        fig.add_trace(go.Scatter(x=history_s.index, y=history_s.values, mode="lines", name="History"))
    if future_s is not None and len(future_s):
        fig.add_trace(go.Scatter(x=future_s.index, y=future_s.values, mode="lines+markers", name="Forecast"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


@st.cache_data(show_spinner=False)
def load_dataframe(file_or_path, sheet_name):
    if file_or_path is None:
        return None

    name = getattr(file_or_path, "name", str(file_or_path))
    ext = (Path(name).suffix or "").lower()

    if ext in [".xlsx", ".xls"]:
        try:
            sn = sheet_name if (sheet_name not in [None, "", " "]) else 0
            return pd.read_excel(file_or_path, sheet_name=sn, engine="openpyxl")
        except ValueError:
            xls = pd.ExcelFile(file_or_path, engine="openpyxl")
            return pd.read_excel(file_or_path, sheet_name=xls.sheet_names[0], engine="openpyxl")
        except Exception:
            return pd.read_excel(file_or_path, sheet_name=0, engine="openpyxl")

    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_or_path, encoding="latin-1")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


# -------------------------------
# Sidebar — Data input
# -------------------------------
st.sidebar.header("1) Load data")
uploaded = st.sidebar.file_uploader("Upload cleaned data (CSV/Excel)", type=["csv", "xlsx"])

default_paths = [
    "cleaned_5min_outlier_free.xlsx",
    "cleaned_1min_outlier_free.xlsx",
]

data_path = None
if uploaded is not None:
    data_path = uploaded
else:
    for p in default_paths:
        if Path(p).exists():
            data_path = p
            break

if data_path is None:
    st.warning("Upload a cleaned dataset (CSV/Excel) or place cleaned_1min_outlier_free.xlsx / cleaned_5min_outlier_free.xlsx in the working directory.")
    st.stop()

sheet = st.sidebar.text_input("Excel sheet name (for .xlsx)", value="cleaned_1min")

# -------------------------------
# Step 1 — Resampling (1–5 minutes)
# -------------------------------
st.sidebar.subheader("Step 1: Resample data")
freq_label = st.sidebar.selectbox(
    "Resample frequency",
    ["1 min", "2 min", "3 min", "4 min", "5 min"],
    index=0,
)
FREQ_MAP = {
    "1 min": "1min",
    "2 min": "2min",
    "3 min": "3min",
    "4 min": "4min",
    "5 min": "5min",
}
freq_code = FREQ_MAP[freq_label]

# -------------------------------
# Step 2 — Hampel cleaning
# -------------------------------
st.sidebar.header("2) Cleaning — Hampel")
do_hampel = st.sidebar.checkbox("Apply Hampel outlier filter", value=True)
hampel_window = st.sidebar.number_input("Hampel window (points)", min_value=3, max_value=201, value=21, step=2)
hampel_sigmas = st.sidebar.number_input("Hampel threshold (sigmas)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

# -------------------------------
# Step 3 — EWMA smoothing
# -------------------------------
st.sidebar.subheader("3) EWMA smoothing")
apply_ewma = st.sidebar.checkbox("Apply EWMA smoothing", value=True)
ewma_alpha = st.sidebar.slider("EWMA alpha (0–1)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# -------------------------------
# Load & prepare dataset
# -------------------------------
df = load_dataframe(data_path, sheet)
if df is None or df.empty:
    st.error("Loaded dataframe is empty. Check your file or sheet name.")
    st.stop()

st.subheader("Raw data (head)")
st.dataframe(df.head(10), use_container_width=True)

# Detect datetime column
dt_col = find_datetime_col(df)
if dt_col:
    # robust parsing for formats like 17/05/25 22.34
    parsed = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
    if parsed.isna().all():
        # try replacing dot in time with colon
        parsed = pd.to_datetime(
            df[dt_col].astype(str).str.replace(r"(\d{2})\.(\d{2})$", r"\1:\2", regex=True),
            errors="coerce",
            dayfirst=True,
        )
    df[dt_col] = parsed
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
else:
    st.warning("No datetime column detected automatically. Resampling may be skipped.")

# Keep only numeric columns
df_num = to_numeric_columns(df.select_dtypes(include=["number", "float", "int", "object"]))
df_num = df_num.select_dtypes(include="number").dropna(how="all")

# ==========================================
# STEP 1 — RESAMPLING
# ==========================================
df_resampled = df_num.copy()
if freq_code:
    if not isinstance(df_resampled.index, pd.DatetimeIndex):
        try:
            df_resampled.index = pd.to_datetime(df_resampled.index, errors="raise")
        except Exception:
            st.warning("Resampling skipped: index is not datetime.")
            freq_code = None

if freq_code:
    df_resampled = df_resampled.resample(freq_code).mean()
    df_resampled = df_resampled.interpolate(method="time", limit_direction="both").ffill().bfill()

st.subheader(f"Step 1 — Resampled data ({freq_code or 'original sampling'})")
st.dataframe(df_resampled.head(12), use_container_width=True)
st.download_button(
    label="Download Step 1 (resampled data)",
    data=df_resampled.to_csv().encode("utf-8"),
    file_name=f"data_step1_resampled_{(freq_code or 'original')}.csv",
    mime="text/csv",
    key="dl_step1",
)

# ==========================================
# STEP 2 — HAMPEL + INTERPOLATION
# ==========================================
df_hampel = df_resampled.copy()
if do_hampel and isinstance(df_hampel.index, pd.DatetimeIndex):
    for c in df_hampel.columns:
        mask = hampel_outliers(df_hampel[c], window=int(hampel_window), n_sigmas=float(hampel_sigmas))
        df_hampel.loc[mask, c] = np.nan
    df_hampel = df_hampel.interpolate(method="time", limit_direction="both").ffill().bfill()

st.subheader("Step 2 — After Hampel cleaning")
st.dataframe(df_hampel.head(12), use_container_width=True)
st.download_button(
    label="Download Step 2 (Hampel-cleaned data)",
    data=df_hampel.to_csv().encode("utf-8"),
    file_name=f"data_step2_hampel_{(freq_code or 'original')}.csv",
    mime="text/csv",
    key="dl_step2",
)

# ==========================================
# STEP 3 — EWMA SMOOTHING
# ==========================================
df_ewma = df_hampel.copy()
if apply_ewma:
    df_ewma[df_ewma.columns] = df_ewma[df_ewma.columns].ewm(alpha=float(ewma_alpha)).mean()

st.subheader("Step 3 — After EWMA smoothing")
st.dataframe(df_ewma.head(12), use_container_width=True)
st.download_button(
    label="Download Step 3 (Hampel + EWMA data)",
    data=df_ewma.to_csv().encode("utf-8"),
    file_name=f"data_step3_hampel_ewma_{(freq_code or 'original')}.csv",
    mime="text/csv",
    key="dl_step3",
)

# Final data used for forecasting
df_num = df_ewma.copy()

# -------------------------------
# Sidebar — Forecast settings
# -------------------------------
st.sidebar.header("4) Forecast settings")

# 4-gas panel mode
use_gas4_mode = st.sidebar.checkbox("Use 4-gas panel mode (CO2, CH4, NH3, H2S)", value=True)
gas_targets = ["CO2", "CH4", "NH3", "H2S"]
available_gas_targets = [c for c in gas_targets if c in df_num.columns]

if use_gas4_mode:
    targets_to_run = available_gas_targets
    if len(targets_to_run) == 0:
        st.error("None of CO2, CH4, NH3, H2S were found in the dataset columns.")
        st.stop()
else:
    target_single = st.sidebar.selectbox("Target column", options=df_num.columns.tolist(), index=0)
    targets_to_run = [target_single]

# Sensible defaults
default_lookback = 288 if (freq_code and str(freq_code).startswith("2")) else 72
lb = st.sidebar.number_input("Lookback (history steps)", min_value=3, max_value=5000, value=int(default_lookback), step=1)
hz = st.sidebar.number_input("Horizon (steps ahead)", min_value=1, max_value=500, value=12, step=1)

val_frac = st.sidebar.slider("Validation fraction", min_value=0.0, max_value=0.4, value=0.15, step=0.05)
test_frac = st.sidebar.slider("Test fraction", min_value=0.05, max_value=0.5, value=0.15, step=0.05)

use_ann = st.sidebar.checkbox("Train ANN (MLPRegressor)", value=True)
use_lstm = st.sidebar.checkbox("Train LSTM (requires TensorFlow)", value=False, disabled=not TF_AVAILABLE)
st.sidebar.caption("TensorFlow available ✅" if TF_AVAILABLE else "TensorFlow not available ❌ — LSTM disabled.")

use_arima = st.sidebar.checkbox("Train ARIMA (statsmodels)", value=False, disabled=not SM_AVAILABLE)
st.sidebar.caption("statsmodels available ✅" if SM_AVAILABLE else "statsmodels not available ❌ — ARIMA disabled.")

with st.sidebar.expander("ARIMA parameters", expanded=use_arima):
    p = st.number_input("AR order p", min_value=0, max_value=10, value=2, step=1, key="arima_p")
    d = st.number_input("Diff order d", min_value=0, max_value=2, value=1, step=1, key="arima_d")
    q = st.number_input("MA order q", min_value=0, max_value=10, value=2, step=1, key="arima_q")

st.sidebar.subheader("Hyperparameter tuning")
#tune_ann = st.sidebar.checkbox("Tune ANN (GridSearchCV)", value=False)
#tune_lstm = st.sidebar.checkbox("Tune LSTM (manual grid)", value=False, disabled=not TF_AVAILABLE)

do_tune_ann = st.sidebar.checkbox("Tune ANN (GridSearchCV)", value=False)
do_tune_lstm = st.sidebar.checkbox("Tune LSTM (manual grid)", value=False, disabled=not TF_AVAILABLE)

with st.sidebar.expander("ANN tuning grid (advanced)", expanded=False):
    ann_grid_small = st.checkbox("Use small grid (faster)", value=True, key="ann_grid_small")
    if ann_grid_small:
        ann_param_grid = {
            "hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 5e-4],
        }
    else:
        ann_param_grid = {
            "hidden_layer_sizes": [(32, 16), (64, 32), (128, 64), (128, 64, 32)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [2e-3, 1e-3, 5e-4],
            "solver": ["adam"],
        }

with st.sidebar.expander("LSTM tuning grid (advanced)", expanded=False):
    lstm_grid_small = st.checkbox("Use small grid (faster)", value=True, key="lstm_grid_small")
    if lstm_grid_small:
        lstm_grid = [
            {"units": 32, "dense_units": 16, "lr": 1e-3, "batch_size": 32},
            {"units": 64, "dense_units": 32, "lr": 1e-3, "batch_size": 32},
            {"units": 64, "dense_units": 32, "lr": 5e-4, "batch_size": 32},
        ]
    else:
        lstm_grid = [
            {"units": u, "dense_units": du, "lr": lr, "batch_size": bs}
            for u in [32, 64, 96]
            for du in [16, 32, 64]
            for lr in [2e-3, 1e-3, 5e-4]
            for bs in [16, 32]
        ]

# -------------------------------
# Charts — Full series split into 4 panels (CO2, CH4, NH3, H2S)
# -------------------------------
st.subheader("Time series (4 panels)")
plot_cols = [c for c in ["CO2", "CH4", "NH3", "H2S"] if c in df_num.columns]
while len(plot_cols) < 4:
    plot_cols.append(None)

pc1, pc2 = st.columns(2)
pc3, pc4 = st.columns(2)
panel_cols = [pc1, pc2, pc3, pc4]

for i, c in enumerate(plot_cols):
    with panel_cols[i]:
        if c is None:
            st.empty()
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_num.index, y=df_num[c], mode="lines", name=str(c)))
        fig.update_layout(title=str(c), height=260, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Multi-target training loop
# -------------------------------
test_preds_by_target = {}      # {"CO2": {"model": "ANN", "data": (...)}}
future_preds_by_target = {}    # {"CO2": {"model": "ANN", "data": pd.Series}}
metrics_rows_all = []
split_info_by_target = {}

for target in targets_to_run:
    st.markdown(f"### Processing target: {target}")

    series = df_num[target].dropna().astype("float32")
    T = len(series)
    N = max(0, T - int(lb) - int(hz) + 1)

    auto_help = ""
    lb_use, hz_use = int(lb), int(hz)

    if N <= 1:
        orig_lb, orig_hz = int(lb), int(hz)
        L = min(orig_lb, max(3, T // 4))
        H = max(1, min(orig_hz, max(1, T // 20)))
        while max(0, T - L - H + 1) <= 1 and L > 3:
            L = max(3, L // 2)
        while max(0, T - L - H + 1) <= 1 and H > 1:
            H = max(1, H // 2)
        lb_use, hz_use = L, H
        N = max(0, T - int(lb_use) - int(hz_use) + 1)
        auto_help = f"⚠️ {target}: short series (T={T}). Auto-adjusted lookback→{lb_use}, horizon→{hz_use} to get N={N}."

    st.write(f"**{target}** → T={T} • N={N} • Lookback={lb_use} • Horizon={hz_use}")
    if auto_help:
        st.info(auto_help)

    if N <= 1:
        st.warning(f"{target}: not enough samples. Skipping.")
        continue

    # Supervised
    X, X_lstm, y, ts = make_supervised(series, int(lb_use), int(hz_use))
    try:
        splits, info = time_split_safe(X, X_lstm, y, ts, val_frac, test_frac)
    except Exception as e:
        st.warning(f"{target}: split failed ({e})")
        continue

    split_info_by_target[target] = info

    (Xtr, Xtr_l, ytr, ttr) = splits["train"]
    (Xva, Xva_l, yva, tva) = splits["val"]
    (Xte, Xte_l, yte, tte) = splits["test"]

    st.write(f"{target} split → train: {info['train']}, val: {info['val']}, test: {info['test']}")

    # Scale
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr_flat = x_scaler.fit_transform(Xtr)
    Xva_flat = x_scaler.transform(Xva) if len(Xva) else Xva
    Xte_flat = x_scaler.transform(Xte) if len(Xte) else Xte

    ytr_s = y_scaler.fit_transform(ytr)
    yva_s = y_scaler.transform(yva) if len(yva) else yva
    yte_s = y_scaler.transform(yte) if len(yte) else yte

    metrics_rows = []
    preds = {}
    future_series_ann = None
    future_series_lstm = None
    future_series_arima = None

    # -------------------------------
    # ANN
    # -------------------------------
    if use_ann:
        try:
            with st.spinner(f"Training ANN ({target})…"):
                fp_ann = fingerprint_series(
                    series,
                    extra={
                        "target": str(target),
                        "freq": str(freq_code),
                        "lb": int(lb_use),
                        "hz": int(hz_use),
                        "val_frac": float(val_frac),
                        "test_frac": float(test_frac),
                        "tune_ann": bool(do_tune_ann),
                        "grid_small": bool(st.session_state.get("ann_grid_small", True)),
                    },
                )

                if do_tune_ann:
                    best_params, ann, best_score = tune_ann_gridsearch(
                        fp_ann, Xtr_flat, ytr_s, ann_param_grid, n_splits=3, random_state=42
                    )
                    st.success(f"{target} ANN tuned ✅ Best params: {best_params} | CV neg-MSE: {best_score:.6f}")
                else:
                    ann = MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=300,
                        random_state=42,
                        early_stopping=(len(Xva_flat) == 0),
                        n_iter_no_change=10,
                        validation_fraction=0.1,
                    )
                    ann.fit(Xtr_flat, ytr_s.ravel())

                def ann_predict(Xflat):
                    y_pred_s = ann.predict(Xflat).reshape(-1, 1)
                    return y_scaler.inverse_transform(y_pred_s)

                # Train metrics
                if len(Xtr_flat):
                    ytr_pred = ann_predict(Xtr_flat)
                    metrics_rows.append({"target": target, "model": "ANN (train)", **compute_metrics(ytr, ytr_pred)})
                    preds["ANN_train"] = (ttr, ytr, ytr_pred)

                # Val/test metrics
                if len(Xva_flat):
                    yva_pred = ann_predict(Xva_flat)
                    metrics_rows.append({"target": target, "model": "ANN (val)", **compute_metrics(yva, yva_pred)})
                    preds["ANN_val"] = (tva, yva, yva_pred)

                if len(Xte_flat):
                    yte_pred = ann_predict(Xte_flat)
                    metrics_rows.append({"target": target, "model": "ANN (test)", **compute_metrics(yte, yte_pred)})
                    preds["ANN_test"] = (tte, yte, yte_pred)

                # Future recursive forecast
                base_series = series.dropna().astype("float32")
                if len(base_series) >= int(lb_use):
                    step_delta = None
                    if freq_code:
                        try:
                            step_delta = pd.to_timedelta(freq_code)
                        except Exception:
                            step_delta = None
                    if step_delta is None and isinstance(base_series.index, pd.DatetimeIndex) and len(base_series.index) >= 2:
                        step_delta = base_series.index[-1] - base_series.index[-2]

                    if step_delta is not None:
                        all_vals = base_series.values.copy()
                        last_time = base_series.index[-1]

                        fut_vals, fut_times = [], []
                        for _ in range(int(hz_use)):
                            window = all_vals[-int(lb_use):].reshape(1, -1)
                            window_scaled = x_scaler.transform(window)
                            next_pred = float(ann_predict(window_scaled).ravel()[0])

                            all_vals = np.append(all_vals, next_pred)
                            last_time = last_time + step_delta
                            fut_times.append(last_time)
                            fut_vals.append(next_pred)

                        future_series_ann = pd.Series(fut_vals, index=pd.DatetimeIndex(fut_times), name=f"{target}_forecast_ann")
        except Exception as e:
            st.warning(f"{target}: ANN training failed ({e})")

    # -------------------------------
    # LSTM
    # -------------------------------
    if use_lstm and TF_AVAILABLE:
        try:
            with st.spinner(f"Training LSTM ({target})…"):
                def scale_seq(Xseq, scaler):
                    n, t, c = Xseq.shape
                    Xf = Xseq.reshape(n, t * c)
                    Xf_s = scaler.transform(Xf)
                    return Xf_s.reshape(n, t, c)

                Xtr_l_s = scale_seq(Xtr_l, x_scaler)
                Xva_l_s = scale_seq(Xva_l, x_scaler) if len(Xva_l) else Xva_l
                Xte_l_s = scale_seq(Xte_l, x_scaler) if len(Xte_l) else Xte_l

                fp_lstm = fingerprint_series(
                    series,
                    extra={
                        "target": str(target),
                        "freq": str(freq_code),
                        "lb": int(lb_use),
                        "hz": int(hz_use),
                        "tune_lstm": bool(do_tune_lstm),
                        "lstm_grid_small": bool(st.session_state.get("lstm_grid_small", True)),
                    },
                )
    

                if do_tune_lstm and len(Xva_l_s):
                    best_params, best_val = tune_lstm_cached(
                        fp_lstm, Xtr_l_s, ytr_s, Xva_l_s, yva_s, lstm_grid, max_epochs=60, patience=8
                    )
                    st.success(f"{target} LSTM tuned ✅ Best params: {best_params} | val_loss: {best_val:.6f}")

                    units = int(best_params["units"])
                    dense_units = int(best_params["dense_units"])
                    lr = float(best_params["lr"])
                    batch_size = int(best_params["batch_size"])
                else:
                    units, dense_units, lr, batch_size = 64, 32, 1e-3, 32
                    if do_tune_lstm and not len(Xva_l_s):
                        st.warning(f"{target}: LSTM tuning requested but validation set is empty → using default params.")

                model = Sequential([
                    LSTM(units, input_shape=(Xtr_l_s.shape[1], Xtr_l_s.shape[2])),
                    Dense(dense_units, activation="relu"),
                    Dense(1),
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

                monitor_metric = "val_loss" if len(Xva_l_s) else "loss"
                es = EarlyStopping(monitor=monitor_metric, patience=10, restore_best_weights=True)

                model.fit(
                    Xtr_l_s, ytr_s,
                    validation_data=(Xva_l_s, yva_s) if len(Xva_l_s) else None,
                    epochs=100,
                    batch_size=batch_size,
                    callbacks=[es],
                    verbose=0,
                )

                if len(Xtr_l_s):
                    ytr_pred_s = model.predict(Xtr_l_s, verbose=0)
                    ytr_pred = y_scaler.inverse_transform(ytr_pred_s)
                    metrics_rows.append({"target": target, "model": "LSTM (train)", **compute_metrics(ytr, ytr_pred)})
                    preds["LSTM_train"] = (ttr, ytr, ytr_pred)

                if len(Xva_l_s):
                    yva_pred_s = model.predict(Xva_l_s, verbose=0)
                    yva_pred = y_scaler.inverse_transform(yva_pred_s)
                    metrics_rows.append({"target": target, "model": "LSTM (val)", **compute_metrics(yva, yva_pred)})
                    preds["LSTM_val"] = (tva, yva, yva_pred)

                if len(Xte_l_s):
                    yte_pred_s = model.predict(Xte_l_s, verbose=0)
                    yte_pred = y_scaler.inverse_transform(yte_pred_s)
                    metrics_rows.append({"target": target, "model": "LSTM (test)", **compute_metrics(yte, yte_pred)})
                    preds["LSTM_test"] = (tte, yte, yte_pred)

                # Future recursive forecast
                base_series = series.dropna().astype("float32")
                if len(base_series) >= int(lb_use):
                    step_delta = None
                    if freq_code:
                        try:
                            step_delta = pd.to_timedelta(freq_code)
                        except Exception:
                            step_delta = None
                    if step_delta is None and isinstance(base_series.index, pd.DatetimeIndex) and len(base_series.index) >= 2:
                        step_delta = base_series.index[-1] - base_series.index[-2]

                    if step_delta is not None:
                        all_vals = base_series.values.copy()
                        last_time = base_series.index[-1]

                        fut_vals, fut_times = [], []
                        for _ in range(int(hz_use)):
                            window = all_vals[-int(lb_use):].reshape(1, -1)
                            window_scaled_flat = x_scaler.transform(window)
                            window_scaled = window_scaled_flat.reshape(1, int(lb_use), 1)

                            next_pred_s = model.predict(window_scaled, verbose=0)
                            next_pred = float(y_scaler.inverse_transform(next_pred_s).ravel()[0])

                            all_vals = np.append(all_vals, next_pred)
                            last_time = last_time + step_delta
                            fut_times.append(last_time)
                            fut_vals.append(next_pred)

                        future_series_lstm = pd.Series(fut_vals, index=pd.DatetimeIndex(fut_times), name=f"{target}_forecast_lstm")
        except Exception as e:
            st.warning(f"{target}: LSTM training failed ({e})")

    # -------------------------------
    # ARIMA
    # -------------------------------
    if use_arima and SM_AVAILABLE:
        import warnings
        try:
            with st.spinner(f"Training ARIMA ({target})…"):
                s_all = series.dropna().astype("float32")

                def split_series_arima(s: pd.Series, val_frac: float, test_frac: float, p_i: int, d_i: int, q_i: int):
                    n = len(s)
                    tf = max(0.05, min(0.5, float(test_frac)))
                    vf = max(0.0, min(0.4, float(val_frac)))

                    n_test = max(1, int(np.floor(tf * n)))
                    n_val = int(np.floor(vf * n))
                    n_train = n - n_val - n_test

                    min_train = max(12, (p_i + d_i + q_i + 2) * 3)

                    while n_train < min_train and n_val > 0:
                        n_val -= 1
                        n_train = n - n_val - n_test
                    while n_train < min_train and n_test > 1:
                        n_test -= 1
                        n_train = n - n_val - n_test

                    if n_train < max(8, p_i + d_i + q_i + 2):
                        raise ValueError(f"Series too short for ARIMA({p_i},{d_i},{q_i}).")

                    tr = s.iloc[:n_train]
                    va = s.iloc[n_train:n_train + n_val]
                    te = s.iloc[n_train + n_val:]
                    return tr, va, te

                p_i, d_i, q_i = int(p), int(d), int(q)
                tr_s, va_s, te_s = split_series_arima(s_all, val_frac, test_frac, p_i, d_i, q_i)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima_res = ARIMA(
                        tr_s,
                        order=(p_i, d_i, q_i),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()

                steps = len(va_s) + len(te_s)
                fc = arima_res.forecast(steps=steps)

                if len(va_s):
                    va_pred = fc.iloc[:len(va_s)]
                    metrics_rows.append({"target": target, "model": "ARIMA (val)", **compute_metrics(va_s.values, va_pred.values)})
                    preds["ARIMA_val"] = (va_s.index, va_s.values.reshape(-1, 1), va_pred.values.reshape(-1, 1))

                if len(te_s):
                    te_pred = fc.iloc[len(va_s):]
                    metrics_rows.append({"target": target, "model": "ARIMA (test)", **compute_metrics(te_s.values, te_pred.values)})
                    preds["ARIMA_test"] = (te_s.index, te_s.values.reshape(-1, 1), te_pred.values.reshape(-1, 1))

                # Future forecast from full series
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima_full = ARIMA(
                        s_all,
                        order=(p_i, d_i, q_i),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()

                step_delta = None
                if freq_code:
                    try:
                        step_delta = pd.to_timedelta(freq_code)
                    except Exception:
                        step_delta = None
                if step_delta is None and isinstance(s_all.index, pd.DatetimeIndex) and len(s_all.index) >= 2:
                    step_delta = s_all.index[-1] - s_all.index[-2]

                if step_delta is not None:
                    fut_steps = int(hz_use)
                    fut_vals = arima_full.forecast(steps=fut_steps).values.astype(float)
                    last_time = s_all.index[-1]
                    fut_times = [last_time + step_delta * (k + 1) for k in range(fut_steps)]
                    future_series_arima = pd.Series(fut_vals, index=pd.DatetimeIndex(fut_times), name=f"{target}_forecast_arima")
        except Exception as e:
            st.warning(f"{target}: ARIMA training failed ({e})")

    # Save metrics
    metrics_rows_all.extend(metrics_rows)

    # Choose one model per target for 4-panel display (priority ANN > LSTM > ARIMA)
    chosen_test = None
    chosen_future = None
    chosen_model_name = None

    if "ANN_test" in preds:
        chosen_test = preds["ANN_test"]
        chosen_future = future_series_ann
        chosen_model_name = "ANN"
    elif "LSTM_test" in preds:
        chosen_test = preds["LSTM_test"]
        chosen_future = future_series_lstm
        chosen_model_name = "LSTM"
    elif "ARIMA_test" in preds:
        chosen_test = preds["ARIMA_test"]
        chosen_future = future_series_arima
        chosen_model_name = "ARIMA"

    if chosen_test is not None:
        test_preds_by_target[target] = {"model": chosen_model_name, "data": chosen_test}
    if chosen_future is not None:
        future_preds_by_target[target] = {"model": chosen_model_name, "data": chosen_future}

# -------------------------------
# Metrics (all targets)
# -------------------------------
st.subheader("Metrics (all targets)")
if len(metrics_rows_all):
    metrics_df = pd.DataFrame(metrics_rows_all)
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("No metrics available. Enable a model and ensure data is sufficient.")

# -------------------------------
# Forecast (Test set) — 4 panels by gas
# -------------------------------
st.subheader("Forecast (Test set) — CO2, CH4, NH3, H2S")
tc1, tc2 = st.columns(2)
tc3, tc4 = st.columns(2)
test_boxes = [tc1, tc2, tc3, tc4]

for i, gas in enumerate(["CO2", "CH4", "NH3", "H2S"]):
    with test_boxes[i]:
        item = test_preds_by_target.get(gas)
        if item is None:
            st.info(f"{gas}: test forecast not available.")
            continue

        model_name = item["model"]
        ts_, y_true_, y_pred_ = item["data"]

        st.plotly_chart(
            plot_forecast(ts_, y_true_, y_pred_, f"{gas} ({model_name}) — Test"),
            use_container_width=True
        )

        df_out = pd.DataFrame({
            "timestamp": ts_,
            "y_true": np.asarray(y_true_).ravel(),
            "y_pred": np.asarray(y_pred_).ravel(),
        })
        st.download_button(
            f"Download {gas} test CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"{gas.lower()}_test_predictions.csv",
            mime="text/csv",
            key=f"dl_test_{gas.lower()}",
        )

# -------------------------------
# Future forecast from last timestamp — 4 panels by gas
# -------------------------------
st.subheader("Future forecast from last timestamp — CO2, CH4, NH3, H2S")
fc1, fc2 = st.columns(2)
fc3, fc4 = st.columns(2)
future_boxes = [fc1, fc2, fc3, fc4]

for i, gas in enumerate(["CO2", "CH4", "NH3", "H2S"]):
    with future_boxes[i]:
        item = future_preds_by_target.get(gas)
        if item is None:
            st.info(f"{gas}: future forecast not available.")
            continue

        model_name = item["model"]
        future_s = item["data"]

        history = df_num[gas].dropna().astype("float32") if gas in df_num.columns else pd.Series(dtype="float32")
        hist_len = min(len(history), int(lb) * 3) if len(history) > 0 else 0
        if hist_len > 0:
            history = history.iloc[-hist_len:]

        st.plotly_chart(
            plot_future_single(history, future_s, f"{gas} ({model_name}) — Future"),
            use_container_width=True
        )

        with st.expander(f"{gas} future forecast table", expanded=False):
            df_future = pd.DataFrame({
                "timestamp": future_s.index,
                f"forecast_{gas.lower()}": future_s.values,
            })
            st.dataframe(df_future, use_container_width=True)

            st.download_button(
                f"Download {gas} future CSV",
                data=df_future.to_csv(index=False).encode("utf-8"),
                file_name=f"future_forecast_{gas.lower()}.csv",
                mime="text/csv",
                key=f"dl_future_{gas.lower()}",
            )

# -------------------------------
# Protocol / reproducibility note
# -------------------------------
st.markdown(
    """
**Time-series split protocol (no leakage):**
- Supervised samples are built chronologically using lookback and horizon.
- Data is split in time order only (train → validation → test), with no shuffling.
- Optional tuning is performed on training data (ANN via TimeSeriesSplit CV; LSTM via validation loss).
- Future forecast is generated recursively from the latest timestamp.
"""
)

st.caption("Tip: For 30 minutes ahead with 2-minute data, set Horizon = 15 steps.")
