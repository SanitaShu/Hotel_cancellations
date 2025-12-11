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
    "MAR": "Morocco", "ABW": "Aruba", "AGO": "Angola", "AIA": "Anguilla",
    "ALB": "Albania", "AND": "Andorra", "ARE": "United Arab Emirates",
    "ARG": "Argentina", "ARM": "Armenia", "ASM": "American Samoa",
    "ATA": "Antarctica", "ATF": "French Southern Territories",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan",
    "BDI": "Burundi", "BEL": "Belgium", "BEN": "Benin", "BFA": "Burkina Faso",
    "BGD": "Bangladesh", "BGR": "Bulgaria", "BHR": "Bahrain",
    "BHS": "Bahamas", "BIH": "Bosnia and Herzegovina", "BLR": "Belarus",
    "BOL": "Bolivia", "BRA": "Brazil", "BRB": "Barbados", "BWA": "Botswana",
    "CAF": "Central African Republic", "CHE": "Switzerland",
    "CHL": "Chile", "CHN": "China", "CIV": "Ivory Coast",
    "CMR": "Cameroon", "CN": "China",
    "COL": "Colombia", "COM": "Comoros", "CPV": "Cape Verde",
    "CRI": "Costa Rica", "CUB": "Cuba", "CYM": "Cayman Islands",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DEU": "Germany",
    "DJI": "Djibouti", "DMA": "Dominica", "DNK": "Denmark",
    "DOM": "Dominican Republic", "DZA": "Algeria", "ECU": "Ecuador",
    "EGY": "Egypt", "ESP": "Spain", "EST": "Estonia", "ETH": "Ethiopia",
    "FIN": "Finland", "FJI": "Fiji", "FRA": "France", "FRO": "Faroe Islands",
    "GAB": "Gabon", "GBR": "United Kingdom", "GEO": "Georgia",
    "GGY": "Guernsey", "GHA": "Ghana", "GIB": "Gibraltar",
    "GLP": "Guadeloupe", "GNB": "Guinea-Bissau", "GRC": "Greece",
    "GTM": "Guatemala", "GUY": "Guyana", "HKG": "Hong Kong",
    "HND": "Honduras", "HRV": "Croatia", "HUN": "Hungary",
    "IDN": "Indonesia", "IMN": "Isle of Man", "IND": "India",
    "IRL": "Ireland", "IRN": "Iran", "IRQ": "Iraq", "ISL": "Iceland",
    "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica", "JEY": "Jersey",
    "JOR": "Jordan", "JPN": "Japan", "KAZ": "Kazakhstan",
    "KEN": "Kenya", "KHM": "Cambodia", "KIR": "Kiribati",
    "KNA": "Saint Kitts and Nevis", "KOR": "South Korea",
    "KWT": "Kuwait", "LAO": "Laos", "LBN": "Lebanon", "LBY": "Libya",
    "LCA": "Saint Lucia", "LIE": "Liechtenstein", "LKA": "Sri Lanka",
    "LTU": "Lithuania", "LUX": "Luxembourg", "LVA": "Latvia",
    "MAC": "Macau", "MCO": "Monaco", "MDG": "Madagascar",
    "MDV": "Maldives", "MEX": "Mexico", "MKD": "North Macedonia",
    "MLI": "Mali", "MLT": "Malta", "MMR": "Myanmar", "MNE": "Montenegro",
    "MOZ": "Mozambique", "MRT": "Mauritania", "MUS": "Mauritius",
    "MWI": "Malawi", "MYS": "Malaysia", "MYT": "Mayotte",
    "NAM": "Namibia", "NCL": "New Caledonia", "NGA": "Nigeria",
    "NIC": "Nicaragua", "NLD": "Netherlands", "NOR": "Norway",
    "NPL": "Nepal", "NZL": "New Zealand", "OMN": "Oman",
    "PAK": "Pakistan", "PAN": "Panama", "PER": "Peru",
    "PHL": "Philippines", "PLW": "Palau", "POL": "Poland",
    "PRI": "Puerto Rico", "PRT": "Portugal", "PRY": "Paraguay",
    "PYF": "French Polynesia", "QAT": "Qatar", "ROU": "Romania",
    "RUS": "Russia", "RWA": "Rwanda", "SAU": "Saudi Arabia",
    "SDN": "Sudan", "SEN": "Senegal", "SGP": "Singapore",
    "SLE": "Sierra Leone", "SLV": "El Salvador", "SMR": "San Marino",
    "SRB": "Serbia", "STP": "Sao Tome and Principe", "SUR": "Suriname",
    "SVK": "Slovakia", "SVN": "Slovenia", "SWE": "Sweden",
    "SYC": "Seychelles", "SYR": "Syria", "TGO": "Togo",
    "THA": "Thailand", "TJK": "Tajikistan", "TMP": "East Timor",
    "TUN": "Tunisia", "TUR": "Turkey", "TWN": "Taiwan",
    "TZA": "Tanzania", "UGA": "Uganda", "UKR": "Ukraine",
    "UMI": "U.S. Minor Outlying Islands", "URY": "Uruguay",
    "USA": "United States", "UZB": "Uzbekistan", "VEN": "Venezuela",
    "VGB": "British Virgin Islands", "VNM": "Vietnam",
    "ZAF": "South Africa", "ZMB": "Zambia", "ZWE": "Zimbabwe",
}

def country_label(code: str) -> str:
    code = str(code)
    if code in ("nan", "None", ""):
        return "Unknown"
    name = COUNTRY_NAME_MAP.get(code)
    if name:
        return f"{name} ({code})"
    return code

def month_name_and_number(series: pd.Series):
    """Return (month_name_series, month_number_series) from arrival_date_month."""
    s = series.dropna().astype(str)
    if s.empty:
        return series.astype(str), pd.Series(index=series.index, dtype=int)

    # numeric
    if pd.api.types.is_numeric_dtype(series):
        nums = series.astype(float).round().astype(int)
        names = nums.map(MONTH_NAME_MAP)
        return names.reindex(series.index), nums.reindex(series.index)

    # try full month names
    try:
        dt = pd.to_datetime(s, format="%B", errors="coerce")
        if dt.notna().any():
            nums = dt.dt.month
            names = nums.map(MONTH_NAME_MAP)
            return names.reindex(series.index), nums.reindex(series.index)
    except Exception:
        pass

    # fallback: treat as categories
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

    if "is_canceled" in df.columns:
        df["is_canceled"] = (
            pd.to_numeric(df["is_canceled"], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    for col in ["adr", "lead_time", "adults", "children", "babies",
                "total_of_special_requests", "total_nights"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# MODEL LOADING (sklearn compatibility patch)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # compatibility patch for _RemainderColsList
    try:
        import sklearn.compose._column_transformer as _ct
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
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
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# basic filters
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

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced filters")

# Lead time range
lead_range = None
if "lead_time" in df.columns:
    lt_min = float(df["lead_time"].min())
    lt_max = float(df["lead_time"].max())
    lead_range = st.sidebar.slider(
        "Lead time (days)",
        min_value=lt_min,
        max_value=lt_max,
        value=(lt_min, lt_max),
    )

# ADR range
adr_range = None
if "adr" in df.columns:
    adr_min = float(df["adr"].min())
    adr_max = float(df["adr"].max())
    adr_range = st.sidebar.slider(
        "Average Daily Rate (ADR)",
        min_value=adr_min,
        max_value=adr_max,
        value=(adr_min, adr_max),
    )

# Total nights range
nights_range = None
if "total_nights" in df.columns:
    tn_min = float(df["total_nights"].min())
    tn_max = float(df["total_nights"].max())
    nights_range = st.sidebar.slider(
        "Total nights (stay length)",
        min_value=tn_min,
        max_value=tn_max,
        value=(tn_min, tn_max),
    )

# Customer type multi-select
cust_selected = None
if "customer_type" in df.columns:
    cust_options = sorted(df["customer_type"].dropna().unique().tolist())
    cust_selected = st.sidebar.multiselect(
        "Customer type",
        options=cust_options,
        default=cust_options,
    )

# Deposit type multi-select
dep_selected = None
if "deposit_type" in df.columns:
    dep_options = sorted(df["deposit_type"].dropna().unique().tolist())
    dep_selected = st.sidebar.multiselect(
        "Deposit type",
        options=dep_options,
        default=dep_options,
    )

# ---- Build mask ----
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

if lead_range and "lead_time" in df.columns:
    lo, hi = lead_range
    mask &= df["lead_time"].between(lo, hi)

if adr_range and "adr" in df.columns:
    lo, hi = adr_range
    mask &= df["adr"].between(lo, hi)

if nights_range and "total_nights" in df.columns:
    lo, hi = nights_range
    mask &= df["total_nights"].between(lo, hi)

if cust_selected and "customer_type" in df.columns:
    mask &= df["customer_type"].isin(cust_selected)

if dep_selected and "deposit_type" in df.columns:
    mask &= df["deposit_type"].isin(dep_selected)

df_f = df.loc[mask].copy()

# ------------------------------------------------------------
# KPI BAR (moved up)
# ------------------------------------------------------------
st.subheader("Dataset overview (after filters)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df_f):,}")
c2.metric("Columns", len(df_f.columns))

if "is_canceled" in df_f.columns and len(df_f):
    c3.metric("Cancellation rate", f"{df_f['is_canceled'].mean() * 100:.1f}%")
else:
    c3.metric("Cancellation rate", "—")

if "adr" in df_f.columns and len(df_f):
    c4.metric("Average ADR", f"{df_f['adr'].mean():.2f}")
else:
    c4.metric("Average ADR", "—")

if "lead_time" in df_f.columns and len(df_f):
    c5.metric("Median lead time (days)", f"{df_f['lead_time'].median():.0f}")
else:
    c5.metric("Median lead time (days)", "—")

st.markdown("---")

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
# PREDICTION UI
# ------------------------------------------------------------
st.subheader("🔮 Cancellation Prediction (manual booking input)")
st.markdown(
    "Fill in the booking details below and let the model predict whether "
    "the booking will be **cancelled** or **not cancelled**."
)

user_vals = defaults.copy()

with st.form("prediction_form"):
    colA, colB = st.columns(2)

    with colA:
        # hotel
        if "hotel" in df.columns and "hotel" in REQUIRED:
            hotel_opts = sorted(df["hotel"].dropna().unique().tolist())
            default_hotel = defaults.get("hotel", hotel_opts[0] if hotel_opts else "")
            if default_hotel not in hotel_opts and hotel_opts:
                default_hotel = hotel_opts[0]
            hotel_choice = st.selectbox("Hotel", hotel_opts, index=hotel_opts.index(default_hotel))
            user_vals["hotel"] = hotel_choice

        # country
        if "country" in df.columns and "country" in REQUIRED:
            codes = sorted(df["country"].dropna().unique().tolist())
            labels = [country_label(c) for c in codes]
            code_to_label = dict(zip(codes, labels))
            label_to_code = {v: k for k, v in code_to_label.items()}
            default_code = defaults.get("country", codes[0] if codes else "")
            default_label = code_to_label.get(default_code, default_code)
            country_label_choice = st.selectbox("Country", labels, index=labels.index(default_label))
            user_vals["country"] = label_to_code.get(country_label_choice, default_code)

        # month
        if "arrival_date_month" in df.columns and "arrival_date_month" in REQUIRED:
            raw_months = sorted(df["arrival_date_month"].dropna().unique().tolist())
            month_labels = []
            for m in raw_months:
                try:
                    val = float(m)
                    label = MONTH_NAME_MAP.get(int(round(val)), str(m))
                except Exception:
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

        # total nights
        if "total_nights" in REQUIRED:
            val = float(defaults.get("total_nights",
                                     df["total_nights"].median() if "total_nights" in df.columns else 2.0))
            user_vals["total_nights"] = st.number_input(
                "Total nights (stay length)",
                min_value=1.0,
                max_value=float(df["total_nights"].max()) if "total_nights" in df.columns else 30.0,
                value=float(val),
                step=1.0,
            )

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

        # special requests
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

if submitted:
    try:
        row_dict = {feat: user_vals.get(feat, defaults.get(feat, 0.0)) for feat in REQUIRED}
        X = pd.DataFrame([row_dict], columns=REQUIRED)

        pred = int(model.predict(X)[0])
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
# VISUALISATIONS
# ------------------------------------------------------------
st.subheader("Visualisations")

# add month columns to filtered df
if "arrival_date_month" in df_f.columns:
    df_f["month_name"], df_f["month_num"] = month_name_and_number(df_f["arrival_date_month"])
    df_f = df_f.dropna(subset=["month_num"])

# --- 1) Combined ADR + Cancellation rate per month
if {"month_num", "month_name", "adr", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    grp = (
        df_f.groupby(["month_num", "month_name"])
        .agg(
            mean_adr=("adr", "mean"),
            cancel_rate=("is_canceled", "mean"),
            count=("is_canceled", "size")
        )
        .reset_index()
        .sort_values("month_num")
    )

    bar = (
        alt.Chart(grp)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("month_name:N", title="Arrival month", sort=list(MONTH_NAME_MAP.values())),
            y=alt.Y("mean_adr:Q", title="Average ADR"),
            tooltip=[
                "month_name",
                alt.Tooltip("mean_adr:Q", format=".1f"),
                alt.Tooltip("cancel_rate:Q", title="Cancellation rate", format=".1%"),
                "count"
            ]
        )
    )

    line = (
        alt.Chart(grp)
        .mark_line(color="#F58518", strokeWidth=3)
        .encode(
            x="month_name:N",
            y=alt.Y("cancel_rate:Q", axis=alt.Axis(title="Cancellation rate", format=".0%")),
        )
    )

    combined_chart = alt.layer(bar, line).resolve_scale(y="independent").properties(
        height=300,
        title="ADR and Cancellation Rate per Month"
    )

    st.altair_chart(combined_chart, use_container_width=True)

# --- 2) Donut chart: cancellations by hotel
if {"hotel", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    hotel_grp = (
        df_f.groupby("hotel")["is_canceled"]
        .mean()
        .reset_index()
        .rename(columns={"is_canceled": "cancel_rate"})
    )

    donut = (
        alt.Chart(hotel_grp)
        .mark_arc(innerRadius=70)
        .encode(
            theta=alt.Theta("cancel_rate:Q", stack=True),
            color=alt.Color("hotel:N", title="Hotel"),
            tooltip=[
                "hotel",
                alt.Tooltip("cancel_rate:Q", title="Cancellation rate", format=".1%")
            ]
        )
        .properties(height=300, title="Cancellation Share by Hotel Type")
    )

    st.altair_chart(donut, use_container_width=True)

# --- 3) Cancellation rate by lead bucket
if {"lead_bucket", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    lead_grp = (
        df_f.groupby("lead_bucket")["is_canceled"]
        .mean()
        .reset_index()
        .sort_values("lead_bucket")
    )

    lead_line = (
        alt.Chart(lead_grp)
        .mark_line(point=True, color="#72B7B2", strokeWidth=3)
        .encode(
            x=alt.X("lead_bucket:N", title="Lead time bucket"),
            y=alt.Y("is_canceled:Q", title="Cancellation rate", axis=alt.Axis(format=".0%")),
            tooltip=["lead_bucket", alt.Tooltip("is_canceled:Q", format=".1%")],
        )
        .properties(height=300, title="Cancellation Rate by Lead Time Bucket")
    )

    st.altair_chart(lead_line, use_container_width=True)

# --- 4) Seasonal decomposition of cancellation rate (full dataset, not filtered)
st.markdown("### Seasonality of cancellations over time")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose

    if {"arrival_date_year", "arrival_date_month", "is_canceled"}.issubset(df.columns):

        df_full = df.copy()
        df_full["month_name"], df_full["month_num"] = month_name_and_number(df_full["arrival_date_month"])
        df_full = df_full.dropna(subset=["month_num"])

        ts = (
            df_full.groupby(["arrival_date_year", "month_num"])
            .agg(cancel_rate=("is_canceled", "mean"))
            .reset_index()
        )

        ts["date"] = pd.to_datetime(
            dict(
                year=ts["arrival_date_year"].astype(int),
                month=ts["month_num"].astype(int),
                day=1,
            )
        )
        ts = ts.sort_values("date")
        series = ts.set_index("date")["cancel_rate"].asfreq("MS").interpolate()

        result = seasonal_decompose(series, model="additive", period=12)

        season_df = pd.DataFrame({
            "date": series.index,
            "observed": series.values,
            "trend": result.trend.values,
            "seasonal": result.seasonal.values,
        }).dropna()

        # observed + trend
        obs_trend = (
            alt.Chart(season_df)
            .transform_fold(
                ["observed", "trend"],
                as_=["type", "value"]
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", axis=alt.Axis(title="Rate", format=".0%")),
                color=alt.Color("type:N", title="Series"),
            )
            .properties(height=250, title="Observed vs Trend (Cancellation Rate)")
        )

        # seasonal
        seasonal_chart = (
            alt.Chart(season_df)
            .mark_line(color="#FF9DA6")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("seasonal:Q", axis=alt.Axis(title="Seasonal component")),
            )
            .properties(height=200, title="Seasonal Component (within year)")
        )

        st.altair_chart(obs_trend, use_container_width=True)
        st.altair_chart(seasonal_chart, use_container_width=True)

    else:
        st.info("Not enough date columns to compute seasonal decomposition.")

except ImportError:
    st.info("Install `statsmodels` in your environment to see seasonal decomposition charts.")

st.markdown("---")

# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------
st.subheader("Sample of filtered data")
st.dataframe(df_f.head(200), use_container_width=True)

st.caption(
    "Model: RandomForest pipeline loaded from rf_booking_pipeline joblib. "
    "Babies are defined as 0–2 years; children as 3–14 years."
)

