import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Hotel Cancellation Analysis", layout="wide")
st.title("Hotel Cancellation Analysis")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
MONTH_NAME_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

COUNTRY_NAME_MAP = {
    # common codes from the hotel_bookings dataset
    "PRT": "Portugal", "GBR": "United Kingdom", "USA": "United States",
    "ESP": "Spain", "IRL": "Ireland", "FRA": "France", "DEU": "Germany",
    "NLD": "Netherlands", "ITA": "Italy", "BEL": "Belgium",
    "BRA": "Brazil", "NOR": "Norway", "SWE": "Sweden", "FIN": "Finland",
    "LVA": "Latvia", "RUS": "Russia", "POL": "Poland", "CHE": "Switzerland",
    # any unknown code will just show as the code itself
}

def country_label(code: str) -> str:
    code = str(code)
    name = COUNTRY_NAME_MAP.get(code)
    if name:
        return f"{name} ({code})"
    return code

def month_name_and_number(series: pd.Series):
    """
    Given a 'arrival_date_month' column which may be numeric or string,
    return (month_name_series, month_number_series) for plotting.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return series.astype(str), pd.Series(index=series.index, dtype=int)

    # Try numeric first
    if pd.api.types.is_numeric_dtype(series):
        nums = series.astype(float).round().astype(int)
        names = nums.map(MONTH_NAME_MAP)
        return names, nums

    # Strings: try full month names
    try:
        dt = pd.to_datetime(s, format="%B", errors="coerce")
        if dt.notna().any():
            nums = dt.dt.month
            names = nums.map(MONTH_NAME_MAP)
            return names.reindex(series.index), nums.reindex(series.index)
    except Exception:
        pass

    # Fallback: treat unique strings as categories in alphabetical order
    cats = s.astype("category")
    nums = cats.cat.codes + 1
    names = s
    return names.reindex(series.index), nums.reindex(series.index)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data
def load_data():
    candidates = [
        "cleaned_output_2025-12-11_16-49-05.parquet",
        "cleaned_output_2025-12-11_11-48-46.parquet",
        "sample_data/cleaned_output_2025-12-11_11-48-46.parquet",
        "cleaned_output_2025-12-11_11-48-46.csv",
        "sample_data/cleaned_output_2025-12-11_11-48-46.csv",
        "df_ready_processed.csv",
        "sample_data/df_ready_processed.csv",
        "hotel_bookings_clean.csv",
        "sample_data/hotel_bookings_clean.csv",
    ]

    df = None
    last_err = None
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
                last_err = e
                st.warning(f"Could not read {path}: {e}")

    if df is None:
        st.error(f"No dataset found. Last error: {last_err}")
        st.stop()

    # basic cleaning
    if "is_canceled" in df.columns:
        df["is_canceled"] = (
            pd.to_numeric(df["is_canceled"], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    for col in ["adr", "lead_time", "adults", "children", "babies",
                "total_of_special_requests"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# MODEL LOADING (with _RemainderColsList patch)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # patch for sklearn version mismatch when unpickling
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
    last_err = None
    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.info(f"Loaded model from **{path}**")
                break
            except Exception as e:
                last_err = e
                st.warning(f"Could not load model {path}: {e}")

    if model is None:
        st.error(f"Trained model (.joblib) could not be loaded. Last error: {last_err}")
        st.stop()

    required = getattr(model, "feature_names_in_", None)
    if required is None:
        st.error("Model has no feature_names_in_. It must be trained on a pandas DataFrame.")
        st.stop()

    return model, list(required)

df = load_data()
model, REQUIRED = load_model()

# ------------------------------------------------------------
# SIDEBAR FILTERS (light)
# ------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# simple filters
if "country" in df.columns:
    country_filter = st.sidebar.selectbox(
        "Country filter",
        ["All"] + sorted(df["country"].dropna().unique().tolist())
    )
else:
    country_filter = "All"

if "hotel" in df.columns:
    hotel_filter = st.sidebar.selectbox(
        "Hotel filter",
        ["All"] + sorted(df["hotel"].dropna().unique().tolist())
    )
else:
    hotel_filter = "All"

if "arrival_date_month" in df.columns:
    month_filter = st.sidebar.selectbox(
        "Arrival month filter",
        ["All"] + sorted(df["arrival_date_month"].dropna().unique().tolist())
    )
else:
    month_filter = "All"

cancel_filter = None
if "is_canceled" in df.columns:
    cancel_filter = st.sidebar.radio(
        "Show bookings",
        ["All", "Canceled", "Not canceled"],
        index=0,
    )

mask = pd.Series(True, index=df.index)
if country_filter != "All" and "country" in df.columns:
    mask &= df["country"] == country_filter
if hotel_filter != "All" and "hotel" in df.columns:
    mask &= df["hotel"] == hotel_filter
if month_filter != "All" and "arrival_date_month" in df.columns:
    mask &= df["arrival_date_month"] == month_filter
if cancel_filter == "Canceled":
    mask &= df["is_canceled"] == 1
elif cancel_filter == "Not canceled":
    mask &= df["is_canceled"] == 0

df_f = df.loc[mask].copy()

# ------------------------------------------------------------
# DEFAULTS FOR MODEL FEATURES
# ------------------------------------------------------------
def build_defaults(df: pd.DataFrame, required):
    defaults = {}
    for feat in required:
        if feat in df.columns:
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                numeric = pd.to_numeric(col, errors="coerce")
                med = numeric.median(skipna=True)
                defaults[feat] = float(med) if pd.notna(med) else 0.0
            else:
                mode = col.mode(dropna=True)
                defaults[feat] = str(mode.iloc[0]) if not mode.empty else ""
        else:
            defaults[feat] = 0.0
    return defaults

defaults = build_defaults(df, REQUIRED)

# ------------------------------------------------------------
# PREDICTION UI (TOP SECTION)
# ------------------------------------------------------------
st.subheader("🔮 Cancellation Prediction (manual booking input)")
st.markdown(
    "Fill in the booking details below and let the model predict whether "
    "the booking will be **cancelled** or **not cancelled**."
)

user_vals = defaults.copy()

with st.form("prediction_form"):
    colA, colB = st.columns(2)

    # ------ Booking / hotel info ------
    with colA:
        # hotel
        if "hotel" in df.columns and "hotel" in REQUIRED:
            hotel_opts = sorted(df["hotel"].dropna().unique().tolist())
            default_hotel = defaults.get("hotel", hotel_opts[0] if hotel_opts else "")
            if default_hotel not in hotel_opts and hotel_opts:
                default_hotel = hotel_opts[0]
            hotel_choice = st.selectbox("Hotel", hotel_opts, index=hotel_opts.index(default_hotel))
            user_vals["hotel"] = hotel_choice

        # country (with full name label)
        if "country" in df.columns and "country" in REQUIRED:
            codes = sorted(df["country"].dropna().unique().tolist())
            labels = [country_label(c) for c in codes]
            code_to_label = dict(zip(codes, labels))
            label_to_code = {v: k for k, v in code_to_label.items()}

            default_code = defaults.get("country", codes[0] if codes else "")
            default_label = code_to_label.get(default_code, default_code)
            country_label_choice = st.selectbox("Country", labels, index=labels.index(default_label))
            user_vals["country"] = label_to_code.get(country_label_choice, default_code)

        # arrival month (names for UI)
        if "arrival_date_month" in df.columns and "arrival_date_month" in REQUIRED:
            raw_months = sorted(df["arrival_date_month"].dropna().unique().tolist())
            # map raw -> label
            month_labels = []
            for m in raw_months:
                try:
                    val = float(m)
                    label = MONTH_NAME_MAP.get(int(round(val)), str(m))
                except Exception:
                    # maybe already a name
                    label = str(m)
                month_labels.append(label)
            raw_to_label = dict(zip(raw_months, month_labels))
            label_to_raw = {v: k for k, v in raw_to_label.items()}

            default_raw = defaults.get("arrival_date_month", raw_months[0] if raw_months else "")
            default_label = raw_to_label.get(default_raw, str(default_raw))
            month_label_choice = st.selectbox(
                "Arrival month",
                list(label_to_raw.keys()),
                index=list(label_to_raw.keys()).index(default_label)
            )
            user_vals["arrival_date_month"] = label_to_raw[month_label_choice]

        # lead time
        if "lead_time" in REQUIRED:
            val = float(defaults.get("lead_time", 7.0))
            user_vals["lead_time"] = st.number_input(
                "Lead time (days)",
                min_value=0.0,
                max_value=float(df["lead_time"].max()) if "lead_time" in df.columns else max(val * 5 + 10, 60.0),
                value=float(val),
                step=1.0,
            )

        # ADR
        if "adr" in REQUIRED:
            val = float(defaults.get("adr", df["adr"].median() if "adr" in df.columns else 100.0))
            user_vals["adr"] = st.number_input(
                "Average Daily Rate (ADR)",
                min_value=0.0,
                max_value=float(df["adr"].max()) if "adr" in df.columns else max(val * 5 + 50, 500.0),
                value=float(val),
                step=1.0,
            )

    # ------ Guest composition & booking conditions ------
    with colB:
        # adults
        if "adults" in REQUIRED:
            val = float(defaults.get("adults", 2.0))
            user_vals["adults"] = st.number_input(
                "Adults",
                min_value=1.0,
                max_value=10.0,
                value=float(val),
                step=1.0,
            )

        # children
        if "children" in REQUIRED:
            val = float(defaults.get("children", 0.0))
            user_vals["children"] = st.number_input(
                "Children (ages 3–14)",
                min_value=0.0,
                max_value=10.0,
                value=float(val),
                step=1.0,
            )

        # babies
        if "babies" in REQUIRED:
            val = float(defaults.get("babies", 0.0))
            user_vals["babies"] = st.number_input(
                "Babies (ages 0–2)",
                min_value=0.0,
                max_value=5.0,
                value=float(val),
                step=1.0,
            )

        # deposit type
        if "deposit_type" in df.columns and "deposit_type" in REQUIRED:
            dep_opts = sorted(df["deposit_type"].dropna().unique().tolist())
            default_dep = defaults.get("deposit_type", dep_opts[0] if dep_opts else "")
            if default_dep not in dep_opts and dep_opts:
                default_dep = dep_opts[0]
            user_vals["deposit_type"] = st.selectbox(
                "Deposit type", dep_opts, index=dep_opts.index(default_dep)
            )

        # customer type
        if "customer_type" in df.columns and "customer_type" in REQUIRED:
            cust_opts = sorted(df["customer_type"].dropna().unique().tolist())
            default_cust = defaults.get("customer_type", cust_opts[0] if cust_opts else "")
            if default_cust not in cust_opts and cust_opts:
                default_cust = cust_opts[0]
            user_vals["customer_type"] = st.selectbox(
                "Customer type", cust_opts, index=cust_opts.index(default_cust)
            )

        # total special requests
        if "total_of_special_requests" in REQUIRED:
            val = float(defaults.get("total_of_special_requests", 0.0))
            user_vals["total_of_special_requests"] = st.number_input(
                "Total special requests",
                min_value=0.0,
                max_value=10.0,
                value=float(val),
                step=1.0,
            )

    st.caption("• Babies: 0–2 years  • Children: 3–14 years")

    submitted = st.form_submit_button("Predict cancellation")

# do prediction
if submitted:
    try:
        row_dict = {feat: user_vals.get(feat, defaults.get(feat, 0.0)) for feat in REQUIRED}
        X = pd.DataFrame([row_dict], columns=REQUIRED)

        pred = int(model.predict(X)[0])
        # assume class 1 == cancelled
        proba_cancel = float(model.predict_proba(X)[0][1])

        if pred == 1:
            st.error(
                f"YES – the booking is likely to be **CANCELLED**.\n\n"
                f"Estimated probability of cancellation: **{proba_cancel:.3f}**"
            )
        else:
            st.success(
                f"NO – the booking is likely to **NOT be cancelled**.\n\n"
                f"Estimated probability of cancellation: **{proba_cancel:.3f}**"
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# ------------------------------------------------------------
# DATASET OVERVIEW (KPIs)
# ------------------------------------------------------------
st.subheader("Dataset overview (after filters)")

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
# VISUALISATIONS (month names)
# ------------------------------------------------------------
st.subheader("Visualisations")

charts = []

# Cancellation rate by arrival month (with month names)
if {"arrival_date_month", "is_canceled"}.issubset(df_f.columns) and len(df_f):
    tmp = df_f[["arrival_date_month", "is_canceled"]].copy()
    tmp["month_name"], tmp["month_num"] = month_name_and_number(tmp["arrival_date_month"])
    tmp = tmp.dropna(subset=["month_num"])
    if len(tmp):
        grp = (
            tmp.groupby(["month_num", "month_name"])["is_canceled"]
            .mean()
            .reset_index()
            .sort_values("month_num")
        )
        ch1 = (
            alt.Chart(grp)
            .mark_bar()
            .encode(
                x=alt.X("month_name:N", title="Arrival month"),
                y=alt.Y("is_canceled:Q", title="Cancellation rate", axis=alt.Axis(format=".0%")),
                tooltip=["month_name", "is_canceled"],
            )
            .properties(height=260)
        )
        charts.append(("Cancellation rate by month", ch1))

# ADR by hotel
if {"hotel", "adr"}.issubset(df_f.columns) and len(df_f):
    grp2 = (
        df_f.groupby("hotel", dropna=False)["adr"]
        .mean()
        .reset_index()
        .sort_values("hotel")
    )
    ch2 = (
        alt.Chart(grp2)
        .mark_bar()
        .encode(
            x=alt.X("hotel:N", title="Hotel"),
            y=alt.Y("adr:Q", title="Average ADR"),
            tooltip=["hotel", "adr"],
        )
        .properties(height=260)
    )
    charts.append(("Average ADR by hotel", ch2))

if charts:
    cols = st.columns(len(charts))
    for col, (title, ch) in zip(cols, charts):
        with col:
            st.markdown(f"### {title}")
            st.altair_chart(ch, use_container_width=True)
else:
    st.info("Not enough columns to draw charts (need arrival_date_month, is_canceled, hotel, adr).")

st.markdown("---")

# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------
st.subheader("Sample of filtered data")
st.dataframe(df_f.head(200), use_container_width=True)

st.caption(
    "Model: RandomForest pipeline loaded from rf_booking_pipeline_2025-12-11_16-53-17.joblib "
    "or rf_booking_pipeline_2025-12-11_11-53-26.joblib. "
    "Babies are defined as 0–2 years; children as 3–14 years."
)
