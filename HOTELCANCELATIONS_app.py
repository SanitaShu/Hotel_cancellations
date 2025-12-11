import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Hotel Cancellation Analysis", layout="wide")
st.title("Hotel Cancellation Analysis")

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the hotel bookings dataset from the most likely paths.
    """
    candidates = [
        # newest parquet first
        "cleaned_output_2025-12-11_16-49-05.parquet",
        "cleaned_output_2025-12-11_11-48-46.parquet",
        "sample_data/cleaned_output_2025-12-11_11-48-46.parquet",
        # CSVs as fallback
        "df_ready_processed.csv",
        "sample_data/df_ready_processed.csv",
        "hotel_bookings_clean.csv",
        "sample_data/hotel_bookings_clean.csv",
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            try:
                if path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                st.info(f"Loaded data from **{path}**")
                break
            except Exception as e:
                st.warning(f"Could not read {path}: {e}")

    if df is None:
        st.error("No dataset found. Please upload a CSV/Parquet next to this app.")
        st.stop()

    # light cleaning
    if "is_canceled" in df.columns:
        df["is_canceled"] = (
            pd.to_numeric(df["is_canceled"], errors="ignore")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    for col in ["adr", "lead_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------
# MODEL LOADING (with _RemainderColsList patch)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the trained RandomForest pipeline.

    We patch sklearn.compose._column_transformer to add the missing
    internal class `_RemainderColsList` so that joblib can unpickle
    models saved with a different scikit-learn version.
    """
    # --- compatibility patch for scikit-learn ---
    try:
        import sklearn.compose._column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Compatibility stub for old ColumnTransformer pickles."""
                pass

            _ct._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
    except Exception as e:
        st.warning(f"Could not apply sklearn compatibility patch: {e}")

    candidates = [
        "rf_booking_pipeline_2025-12-11_16-53-17.joblib",
        "rf_booking_pipeline_2025-12-11_11-53-26.joblib",
        "sample_data/rf_booking_pipeline_2025-12-11_11-53-26.joblib",
    ]

    model = None
    last_error = None

    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.info(f"Loaded model from **{path}**")
                break
            except Exception as e:
                last_error = e
                st.warning(f"Could not load model {path}: {e}")

    if model is None:
        st.error(
            "Trained model (.joblib) could not be loaded.\n\n"
            f"Last error: {last_error}"
        )
        st.stop()

    required = getattr(model, "feature_names_in_", None)
    if required is None:
        st.error(
            "Model has no feature_names_in_. "
            "It must be trained on a pandas DataFrame."
        )
    return model, list(required)


df = load_data()
model, REQUIRED = load_model()

# ------------------------------------------------------------
# SIDEBAR – LIGHT FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Simple key filters only
if "country" in df.columns:
    country = st.sidebar.selectbox(
        "Country", ["All"] + sorted(df["country"].dropna().unique().tolist())
    )
else:
    country = "All"

if "hotel" in df.columns:
    hotel = st.sidebar.selectbox(
        "Hotel", ["All"] + sorted(df["hotel"].dropna().unique().tolist())
    )
else:
    hotel = "All"

if "arrival_date_month" in df.columns:
    month = st.sidebar.selectbox(
        "Arrival month", ["All"] + sorted(df["arrival_date_month"].dropna().unique().tolist())
    )
else:
    month = "All"

# Lead time & ADR sliders (if present)
lead_range = None
adr_range = None

if "lead_time" in df.columns:
    lt_min, lt_max = float(df["lead_time"].min()), float(df["lead_time"].max())
    lead_range = st.sidebar.slider(
        "Lead time (days)",
        min_value=lt_min,
        max_value=lt_max,
        value=(lt_min, lt_max),
    )

if "adr" in df.columns:
    adr_min, adr_max = float(df["adr"].min()), float(df["adr"].max())
    adr_range = st.sidebar.slider(
        "Average Daily Rate (ADR)",
        min_value=adr_min,
        max_value=adr_max,
        value=(adr_min, adr_max),
    )

# Cancellation status filter
cancel_filter = None
if "is_canceled" in df.columns:
    cancel_filter = st.sidebar.radio(
        "Cancellation status", ["All", "Canceled", "Not canceled"], index=0
    )

# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------
mask = pd.Series(True, index=df.index)

if country != "All" and "country" in df.columns:
    mask &= df["country"] == country

if hotel != "All" and "hotel" in df.columns:
    mask &= df["hotel"] == hotel

if month != "All" and "arrival_date_month" in df.columns:
    mask &= df["arrival_date_month"] == month

if lead_range and "lead_time" in df.columns:
    lo, hi = lead_range
    mask &= df["lead_time"].between(lo, hi)

if adr_range and "adr" in df.columns:
    lo, hi = adr_range
    mask &= df["adr"].between(lo, hi)

if cancel_filter == "Canceled":
    mask &= df["is_canceled"] == 1
elif cancel_filter == "Not canceled":
    mask &= df["is_canceled"] == 0

df_f = df.loc[mask].copy()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", len(df_f.columns))

if "is_canceled" in df_f.columns and len(df_f):
    c3.metric("Cancellation rate", f"{df_f['is_canceled'].mean() * 100:.1f}%")
else:
    c3.metric("Cancellation rate", "—")

if "adr" in df_f.columns and len(df_f):
    c4.metric("Average ADR", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average ADR", "—")

st.markdown("---")

# ------------------------------------------------------------
# VISUALS (SIMPLE)
# ------------------------------------------------------------
st.subheader("Visualisations")

charts = []

# Cancellation rate by month
if {"arrival_date_month", "is_canceled"}.issubset(df_f.columns) and len(df_f):
    tmp = df_f.copy()
    try:
        tmp["_month_num"] = pd.to_datetime(
            tmp["arrival_date_month"], format="%B"
        ).dt.month
    except Exception:
        tmp["_month_num"] = pd.factorize(tmp["arrival_date_month"])[0] + 1

    tmp = (
        tmp.groupby(["_month_num", "arrival_date_month"])["is_canceled"]
        .mean()
        .reset_index()
        .sort_values("_month_num")
    )

    ch1 = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("arrival_date_month:N", title="Month"),
            y=alt.Y(
                "is_canceled:Q",
                title="Cancellation rate",
                axis=alt.Axis(format=".0%")
            ),
            tooltip=["arrival_date_month", "is_canceled"],
        )
        .properties(height=260)
    )
    charts.append(("Cancellation Rate by Month", ch1))

# ADR by hotel
if {"hotel", "adr"}.issubset(df_f.columns) and len(df_f):
    tmp2 = (
        df_f.groupby("hotel", dropna=False)["adr"]
        .mean()
        .reset_index()
        .sort_values("hotel")
    )
    ch2 = (
        alt.Chart(tmp2)
        .mark_bar()
        .encode(
            x=alt.X("hotel:N", title="Hotel"),
            y=alt.Y("adr:Q", title="Average ADR"),
            tooltip=["hotel", "adr"],
        )
        .properties(height=260)
    )
    charts.append(("Average ADR by Hotel", ch2))

if charts:
    cols = st.columns(len(charts))
    for col, (title, ch) in zip(cols, charts):
        with col:
            st.markdown(f"### {title}")
            st.altair_chart(ch, use_container_width=True)
else:
    st.info(
        "Not enough data columns to draw charts "
        "(need arrival_date_month, is_canceled, hotel, adr)."
    )

st.markdown("---")

# ------------------------------------------------------------
# SIMPLE DATA TABLE
# ------------------------------------------------------------
st.subheader("Sample of filtered data")
st.dataframe(df_f.head(200), use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# ML PREDICTION SECTION – MANUAL FEATURE ENTRY
# ------------------------------------------------------------
st.subheader("🔮 Cancellation Prediction (manual booking input)")

st.markdown(
    "Fill in the booking details below and let the model predict whether "
    "the booking will be **cancelled** or **not cancelled**."
)

# Build sensible defaults for ALL required features
defaults = {}
for feat in REQUIRED:
    if feat in df.columns:
        col = df[feat]
        if col.dtype == "object":
            mode = col.mode(dropna=True)
            defaults[feat] = mode.iloc[0] if not mode.empty else ""
        else:
            defaults[feat] = float(col.median())
    else:
        defaults[feat] = 0.0

# Define a smaller set of features we expose to the user
important_feats = [
    "hotel",
    "country",
    "arrival_date_month",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "adr",
    "deposit_type",
    "customer_type",
    "market_segment",
    "distribution_channel",
    "total_of_special_requests",
]

with st.form("manual_prediction_form"):
    cols1 = st.columns(3)

    user_vals = defaults.copy()

    # CATEGORICAL SELECTBOXES
    def cat_input(colname, label, column_group):
        if colname in REQUIRED and colname in df.columns:
            opts = sorted(df[colname].dropna().astype(str).unique())
            default_val = (
                defaults[colname]
                if defaults[colname] in opts
                else (opts[0] if opts else "")
            )
            user_vals[colname] = column_group.selectbox(
                label, opts, index=opts.index(default_val) if default_val in opts else 0
            )

    # NUMERIC INPUT
    def num_input(colname, label, column_group, minv=0.0, maxv=None, step=1.0):
        if colname in REQUIRED:
            val = float(defaults.get(colname, 0.0))
            if maxv is None:
                maxv = float(df[colname].max()) if colname in df.columns else val * 5 + 10
            user_vals[colname] = column_group.number_input(
                label, min_value=float(minv), max_value=float(maxv), value=float(val), step=step
            )

    # First column: hotel + country + month
    cat_input("hotel", "Hotel", cols1[0])
    cat_input("country", "Country", cols1[0])
    cat_input("arrival_date_month", "Arrival month", cols1[0])

    # Second column: lead time, stays_in_*_nights
    if "lead_time" in REQUIRED:
        num_input("lead_time", "Lead time (days)", cols1[1], minv=0.0, step=1.0)
    if "stays_in_weekend_nights" in REQUIRED:
        num_input("stays_in_weekend_nights", "Weekend nights", cols1[1], minv=0.0, step=1.0)
    if "stays_in_week_nights" in REQUIRED:
        num_input("stays_in_week_nights", "Week nights", cols1[1], minv=0.0, step=1.0)

    # Third column: guests + ADR
    if "adults" in REQUIRED:
        num_input("adults", "Adults", cols1[2], minv=1.0, step=1.0)
    if "children" in REQUIRED:
        num_input("children", "Children", cols1[2], minv=0.0, step=1.0)
    if "babies" in REQUIRED:
        num_input("babies", "Babies", cols1[2], minv=0.0, step=1.0)
    if "adr" in REQUIRED:
        num_input("adr", "Average Daily Rate (ADR)", cols1[2], minv=0.0, step=1.0)

    st.markdown("----")
    cols2 = st.columns(3)

    cat_input("deposit_type", "Deposit type", cols2[0])
    cat_input("customer_type", "Customer type", cols2[0])
    cat_input("market_segment", "Market segment", cols2[1])
    cat_input("distribution_channel", "Distribution channel", cols2[1])

    if "total_of_special_requests" in REQUIRED:
        num_input(
            "total_of_special_requests",
            "Total special requests",
            cols2[2],
            minv=0.0,
            step=1.0,
        )

    submitted = st.form_submit_button("Predict cancellation")

if submitted:
    try:
        # Build full feature dict in REQUIRED order
        row_dict = {feat: user_vals.get(feat, defaults.get(feat, 0.0)) for feat in REQUIRED}
        X = pd.DataFrame([row_dict], columns=REQUIRED)

        pred = int(model.predict(X)[0])
        proba_cancel = float(model.predict_proba(X)[0][1])  # probability of cancellation

        if pred == 1:
            label = "YES – the booking is likely to be **CANCELLED**."
        else:
            label = "NO – the booking is likely to **NOT be cancelled**."

        st.success(label)
        st.write(f"Estimated probability of cancellation: **{proba_cancel:.3f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Model: RandomForest pipeline loaded from rf_booking_pipeline_2025-12-11_16-53-17.joblib (or fallback).")
