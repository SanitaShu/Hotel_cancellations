import os
import calendar
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
# COUNTRY CODE → LABEL HELPER
# ------------------------------------------------------------
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

    for col in [
        "adr",
        "lead_time",
        "adults",
        "children",
        "babies",
        "total_of_special_requests",
        "total_nights",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # Patch for old sklearn column transformer internals
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
        st.error("Model has no feature_names_in_. Train it on a pandas DataFrame.")
        st.stop()

    return model, list(required)


df = load_data()
model, REQUIRED = load_model()

# ------------------------------------------------------------
# DEFAULT VALUES FOR FEATURES
# ------------------------------------------------------------
def build_defaults(df: pd.DataFrame, required_cols):
    defaults = {}
    for feat in required_cols:
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
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reset filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if "country" in df.columns:
    country_filter = st.sidebar.selectbox(
        "Country filter",
        ["All"] + sorted(df["country"].dropna().unique().tolist()),
    )
else:
    country_filter = "All"

if "hotel" in df.columns:
    hotel_filter = st.sidebar.selectbox(
        "Hotel filter",
        ["All"] + sorted(df["hotel"].dropna().unique().tolist()),
    )
else:
    hotel_filter = "All"

if "arrival_date_month" in df.columns:
    month_filter = st.sidebar.selectbox(
        "Arrival month filter",
        ["All"] + sorted(df["arrival_date_month"].dropna().unique().tolist()),
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

cust_selected = None
if "customer_type" in df.columns:
    cust_options = sorted(df["customer_type"].dropna().unique().tolist())
    cust_selected = st.sidebar.multiselect(
        "Customer type",
        options=cust_options,
        default=cust_options,
    )

dep_selected = None
if "deposit_type" in df.columns:
    dep_options = sorted(df["deposit_type"].dropna().unique().tolist())
    dep_selected = st.sidebar.multiselect(
        "Deposit type",
        options=dep_options,
        default=dep_options,
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
# KPI BAR
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
# PREDICTION UI
# ------------------------------------------------------------
st.subheader("🔮 Cancellation Prediction (manual booking input)")
st.markdown(
    "Fill in the booking details below and let the model predict whether "
    "the booking will be **cancelled** or **not cancelled**."
)

def build_defaults_row():
    return defaults.copy()

user_vals = build_defaults_row()

with st.form("prediction_form"):
    colA, colB = st.columns(2)

    # LEFT
    with colA:
        if "hotel" in df.columns and "hotel" in REQUIRED:
            hotel_opts = sorted(df["hotel"].dropna().unique().tolist())
            default_hotel = defaults.get("hotel", hotel_opts[0] if hotel_opts else "")
            if default_hotel not in hotel_opts and hotel_opts:
                default_hotel = hotel_opts[0]
            user_vals["hotel"] = st.selectbox(
                "Hotel", hotel_opts, index=hotel_opts.index(default_hotel)
            )

        if "country" in df.columns and "country" in REQUIRED:
            codes = sorted(df["country"].dropna().unique().tolist())
            labels = [country_label(c) for c in codes]
            code_to_label = dict(zip(codes, labels))
            label_to_code = {v: k for k, v in code_to_label.items()}
            default_code = defaults.get("country", codes[0] if codes else "")
            default_label = code_to_label.get(default_code, default_code)
            label_choice = st.selectbox(
                "Country", labels, index=labels.index(default_label)
            )
            user_vals["country"] = label_to_code.get(label_choice, default_code)

        if "arrival_date_month" in REQUIRED:
            if "arrival_date_month" in df.columns:
                nums = sorted(
                    pd.to_numeric(
                        df["arrival_date_month"], errors="coerce"
                    ).dropna().unique()
                )
            else:
                nums = list(range(1, 13))

            labels = [calendar.month_name[int(n)] for n in nums]
            num_to_label = dict(zip(nums, labels))
            label_to_num = {v: k for k, v in num_to_label.items()}
            default_num = float(
                defaults.get("arrival_date_month", nums[0] if nums else 1)
            )
            default_label = num_to_label.get(default_num, labels[0])
            chosen_label = st.selectbox(
                "Arrival month", labels, index=labels.index(default_label)
            )
            user_vals["arrival_date_month"] = label_to_num[chosen_label]

        if "lead_time" in REQUIRED:
            val = float(defaults.get("lead_time", 7.0))
            max_lt = float(df["lead_time"].max()) if "lead_time" in df.columns else 365.0
            user_vals["lead_time"] = st.number_input(
                "Lead time (days)",
                min_value=0.0,
                max_value=max_lt,
                value=float(val),
                step=1.0,
            )

        if "adr" in REQUIRED:
            val = float(
                defaults.get(
                    "adr", df["adr"].median() if "adr" in df.columns else 100.0
                )
            )
            max_adr = float(df["adr"].max()) if "adr" in df.columns else 500.0
            user_vals["adr"] = st.number_input(
                "Average Daily Rate (ADR)",
                min_value=0.0,
                max_value=max_adr,
                value=float(val),
                step=1.0,
            )

        if "total_nights" in REQUIRED:
            val = float(
                defaults.get(
                    "total_nights",
                    df["total_nights"].median()
                    if "total_nights" in df.columns
                    else 2.0,
                )
            )
            max_nights = (
                float(df["total_nights"].max())
                if "total_nights" in df.columns
                else 30.0
            )
            user_vals["total_nights"] = st.number_input(
                "Total nights (stay length)",
                min_value=1.0,
                max_value=max_nights,
                value=float(val),
                step=1.0,
            )

    # RIGHT
    with colB:
        if "adults" in REQUIRED:
            val = float(defaults.get("adults", 2.0))
            user_vals["adults"] = st.number_input(
                "Adults",
                min_value=1.0,
                max_value=10.0,
                value=float(val),
                step=1.0,
            )

        if "children" in REQUIRED:
            val = float(defaults.get("children", 0.0))
            user_vals["children"] = st.number_input(
                "Children (ages 3–14)",
                min_value=0.0,
                max_value=10.0,
                value=float(val),
                step=1.0,
            )

        if "babies" in REQUIRED:
            val = float(defaults.get("babies", 0.0))
            user_vals["babies"] = st.number_input(
                "Babies (ages 0–2)",
                min_value=0.0,
                max_value=5.0,
                value=float(val),
                step=1.0,
            )

        if "deposit_type" in df.columns and "deposit_type" in REQUIRED:
            dep_opts = sorted(df["deposit_type"].dropna().unique().tolist())
            default_dep = defaults.get("deposit_type", dep_opts[0] if dep_opts else "")
            if default_dep not in dep_opts and dep_opts:
                default_dep = dep_opts[0]
            user_vals["deposit_type"] = st.selectbox(
                "Deposit type", dep_opts, index=dep_opts.index(default_dep)
            )

        if "customer_type" in df.columns and "customer_type" in REQUIRED:
            cust_opts = sorted(df["customer_type"].dropna().unique().tolist())
            default_cust = defaults.get(
                "customer_type", cust_opts[0] if cust_opts else ""
            )
            if default_cust not in cust_opts and cust_opts:
                default_cust = cust_opts[0]
            user_vals["customer_type"] = st.selectbox(
                "Customer type", cust_opts, index=cust_opts.index(default_cust)
            )

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

def month_name_and_number(month_col):
    month_num = pd.to_numeric(month_col, errors="coerce").astype("Int64")
    month_name = month_num.apply(
        lambda x: calendar.month_name[int(x)] if pd.notnull(x) else None
    )
    return month_name, month_num

if "arrival_date_month" in df_f.columns:
    df_f["month_name"], df_f["month_num"] = month_name_and_number(
        df_f["arrival_date_month"]
    )
    df_f = df_f.dropna(subset=["month_num"])

# 1) ADR + cancellation per month
if {"month_name", "month_num", "adr", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    grp = (
        df_f.groupby(["month_num", "month_name"])
        .agg(
            mean_adr=("adr", "mean"),
            cancel_rate=("is_canceled", "mean"),
            count=("is_canceled", "size"),
        )
        .reset_index()
        .sort_values("month_num")
    )

    bar = (
        alt.Chart(grp)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X(
                "month_name:N",
                title="Arrival month",
                sort=alt.SortField(field="month_num", order="ascending"),
            ),
            y=alt.Y("mean_adr:Q", title="Average ADR (€)"),
            tooltip=[
                "month_name",
                alt.Tooltip("mean_adr:Q", format=".1f"),
                alt.Tooltip("cancel_rate:Q", title="Cancellation rate", format=".1%"),
                "count",
            ],
        )
    )

    line = (
        alt.Chart(grp)
        .mark_line(color="#F58518", strokeWidth=3)
        .encode(
            x=alt.X(
                "month_name:N",
                sort=alt.SortField(field="month_num", order="ascending"),
            ),
            y=alt.Y(
                "cancel_rate:Q",
                axis=alt.Axis(title="Cancellation rate", format=".0%"),
            ),
        )
    )

    combo = (
        alt.layer(bar, line)
        .resolve_scale(y="independent")
        .properties(height=300, title="ADR and Cancellation Rate per Month")
    )

    st.altair_chart(combo, use_container_width=True)

# 2) Donut chart – share of ALL cancellations by hotel type (sums to 100%)
if {"hotel", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    # Consider only cancelled bookings
    canceled_df = df_f[df_f["is_canceled"] == 1]

    # If there are no cancellations in the filtered data, show a note
    if len(canceled_df) == 0:
        st.info("No cancelled bookings in the current filters — donut chart skipped.")
    else:
        # Count cancellations by hotel and convert to percentage of total
        hotel_grp = (
            canceled_df.groupby("hotel")["is_canceled"]
            .count()
            .reset_index()
            .rename(columns={"is_canceled": "cancel_count"})
        )

        total_cancel = hotel_grp["cancel_count"].sum()
        hotel_grp["cancel_pct"] = hotel_grp["cancel_count"] / total_cancel * 100

        # Label text: e.g. "City Hotel – 56.7%"
        hotel_grp["label"] = (
            hotel_grp["hotel"]
            + " – "
            + hotel_grp["cancel_pct"].round(1).astype(str)
            + "%"
        )

        base = alt.Chart(hotel_grp)

        # Donut itself
        donut = (
            base.mark_arc(innerRadius=60, outerRadius=110)
            .encode(
                theta=alt.Theta("cancel_pct:Q", stack=True),
                color=alt.Color(
                    "hotel:N",
                    title="Hotel",
                    scale=alt.Scale(scheme="tableau20"),
                ),
                tooltip=[
                    "hotel",
                    alt.Tooltip(
                        "cancel_count:Q",
                        title="Number of cancelled bookings",
                        format=",.0f",
                    ),
                    alt.Tooltip(
                        "cancel_pct:Q",
                        title="Share of all cancellations (%)",
                        format=".1f",
                    ),
                ],
            )
        )

        # Labels outside the donut
        text = (
            base.mark_text(radius=140, size=14, fontWeight="bold")
            .encode(
                theta=alt.Theta("cancel_pct:Q", stack=True),
                text="label:N",
                color=alt.value("black"),
            )
        )

        pie_final = (donut + text).properties(
            height=350,
            title="Share of All Cancellations by Hotel Type",
        )

        st.altair_chart(pie_final, use_container_width=True)


# 3) Lead bucket line chart – only buckets with data, in logical order
if {"lead_bucket", "is_canceled"}.issubset(df_f.columns) and len(df_f):

    # Logical order for buckets
    lead_order = ["<=7d", "8-30d", "31-90d", "91-180d", "181-365d", ">365d"]
    order_map = {b: i for i, b in enumerate(lead_order)}

    # Aggregate only buckets that actually appear in the filtered data
    lead_grp = (
        df_f.groupby("lead_bucket")["is_canceled"]
        .agg(cancel_rate="mean", count="size")
        .reset_index()
    )

    # Map to order index and keep only known buckets
    lead_grp["bucket_order"] = lead_grp["lead_bucket"].map(order_map)
    lead_grp = lead_grp.dropna(subset=["bucket_order"]).sort_values("bucket_order")

    # If after all filters we somehow have no valid buckets, skip the chart
    if lead_grp.empty:
        st.info("No lead-time buckets available for the current filters.")
    else:
        lead_line = (
            alt.Chart(lead_grp)
            .mark_line(point=True, color="#72B7B2", strokeWidth=3)
            .encode(
                x=alt.X(
                    "lead_bucket:N",
                    title="Lead time bucket",
                    sort=alt.SortField(
                        field="bucket_order", order="ascending"
                    ),
                ),
                y=alt.Y(
                    "cancel_rate:Q",
                    title="Cancellation rate",
                    axis=alt.Axis(format=".0%"),
                ),
                tooltip=[
                    "lead_bucket",
                    alt.Tooltip(
                        "cancel_rate:Q",
                        format=".1%",
                        title="Cancellation rate",
                    ),
                    alt.Tooltip("count:Q", format=",.0f", title="Number of bookings"),
                ],
            )
            .properties(
                height=300,
                title="Cancellation Rate by Lead Time Bucket",
            )
        )

        st.altair_chart(lead_line, use_container_width=True)
# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------
st.subheader("Sample of filtered data")
st.dataframe(df_f.head(200), use_container_width=True)

st.caption(
    "Model: RandomForest pipeline. Babies: 0–2 years; Children: 3–14 years."
)
