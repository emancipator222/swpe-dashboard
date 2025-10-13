import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import re
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Callable, Optional, Tuple


st.set_page_config(
    page_title="SWPE Dashboard Final",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_STYLE = """
<style>
:root,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"] {
    color-scheme: dark;
}
[data-testid="stAppViewContainer"] {
    background: #0f1116;
}
[data-testid="stAppViewContainer"] .block-container {
    padding: 2.5rem 2.5rem 3rem 2.5rem;
}
[data-testid="stAppViewContainer"] * {
    color: #e7ecff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121623 0%, #0d0f18 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #f5f7ff !important;
    font-weight: 500;
}
[data-testid="stSidebar"] label {
    font-weight: 600;
    color: #cdd7ff !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stDateInput input,
[data-testid="stSidebar"] .stFileUploader,
[data-testid="stSidebar"] .stMultiSelect,
[data-testid="stSidebar"] .stSlider {
    background: rgba(30, 35, 52, 0.85);
    border-radius: 10px;
    border: 1px solid rgba(90, 112, 255, 0.25);
    color: #f5f7ff !important;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    border-radius: 12px;
    border: none;
    padding: 0.65rem 1.6rem;
    background: linear-gradient(120deg, #5b7bff, #3ad6ff);
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 12px 25px rgba(46, 134, 255, 0.35);
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    opacity: 0.92;
}
[data-testid="stHeader"] {
    background: rgba(12, 14, 20, 0.75);
    backdrop-filter: blur(6px);
}
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(30, 36, 54, 0.92), rgba(18, 22, 35, 0.92));
    border-radius: 16px;
    border: 1px solid rgba(86, 120, 255, 0.26);
    padding: 14px 18px;
    box-shadow: 0 18px 40px rgba(12, 19, 36, 0.55);
}
.stDataFrame {
    border: 1px solid rgba(86, 120, 255, 0.18);
    border-radius: 14px;
    overflow: hidden;
}
div[data-testid="stVerticalBlock"] > div:has(.stDataFrame) {
    background: rgba(20, 24, 36, 0.7);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(89, 150, 255, 0.18);
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #f5f7ff;
}
.stMarkdown a {
    color: #55c0ff;
}
.stRadio > div, .stSelectbox > div {
    background: rgba(23, 28, 42, 0.85);
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(98, 120, 255, 0.18);
}
.swpe-sticky-marker { display: none; }
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) {
    position: sticky;
    top: 64px;
    z-index: 40;
    background: rgba(15, 17, 25, 0.9);
    border: 1px solid rgba(90, 112, 255, 0.18);
    border-radius: 18px;
    padding: 0.75rem 1.1rem 0.9rem 1.1rem;
    box-shadow: 0 22px 48px rgba(8, 12, 25, 0.55);
    backdrop-filter: blur(10px);
    margin-bottom: 1.6rem;
}
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) .stDateInput > div > div > input {
    background: rgba(24, 29, 44, 0.85);
    border-radius: 12px;
    border: 1px solid rgba(90, 112, 255, 0.16);
}
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) label {
    font-weight: 600;
    color: #d6dcff;
}
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) .stRadio > div {
    background: transparent;
    border: none;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) .stRadio > div > label {
    background: rgba(24, 29, 44, 0.85);
    padding: 0.45rem 0.9rem;
    border-radius: 10px;
    border: 1px solid rgba(88, 113, 255, 0.2);
    margin-right: 0.35rem;
    display: inline-flex;
    align-items: center;
}
div[data-testid="stVerticalBlock"] > div:has(> div.swpe-sticky-marker) .stRadio > div > label[data-baseweb="radio"] {
    margin-bottom: 0;
}
[data-testid="stTabs"] > div > div > div {
    background: rgba(14, 17, 26, 0.92);
    border-radius: 18px;
    padding: 0.45rem;
    border: 1px solid rgba(86, 120, 255, 0.18);
}
[data-baseweb="tab-list"] {
    gap: 0.4rem;
}
button[role="tab"] {
    border-radius: 13px;
    padding: 0.6rem 1.35rem;
    background: rgba(27, 32, 48, 0.7);
    color: #cfd5ff;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    font-weight: 600;
}
button[role="tab"]:hover {
    border-color: rgba(90, 122, 255, 0.4);
    color: #f1f3ff;
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(125deg, rgba(91, 123, 255, 0.85), rgba(58, 214, 255, 0.78));
    color: #ffffff;
    box-shadow: 0 12px 26px rgba(54, 105, 255, 0.45);
}
.swpe-kpi-card {
    background: linear-gradient(135deg, rgba(30, 36, 54, 0.95), rgba(15, 18, 31, 0.95));
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(88, 122, 255, 0.2);
    box-shadow: 0 24px 44px rgba(9, 13, 28, 0.6);
    width: 100%;
}
.swpe-kpi-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #cdd7ff;
    margin-bottom: 0.25rem;
}
.swpe-kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f5f7ff;
}
.swpe-kpi-delta {
    margin-top: 0.35rem;
    font-size: 0.84rem;
    font-weight: 500;
    color: #8ca2ff;
}
.swpe-kpi-delta span {
    display: inline-block;
    margin-right: 0.6rem;
}
.swpe-kpi-delta .negative {
    color: #ff7b9b;
}
.swpe-kpi-delta .positive {
    color: #6fe3a6;
}
.block-container .stTextInput > div > div > input,
.block-container .stDateInput > div > div > input,
.block-container .stNumberInput > div > div > input {
    background: rgba(24, 29, 44, 0.85);
    border-radius: 12px;
    border: 1px solid rgba(90, 112, 255, 0.18);
    color: #f5f7ff !important;
}
.block-container .stTextInput > div > div > input::placeholder,
.block-container .stDateInput > div > div > input::placeholder {
    color: rgba(207, 213, 255, 0.65);
}
.swpe-chart-controls {
    background: rgba(16, 20, 32, 0.78);
    border-radius: 16px;
    border: 1px solid rgba(86, 120, 255, 0.16);
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.2rem;
}
.swpe-chart-controls label {
    color: #dae1ff !important;
    font-weight: 600;
}
</style>
"""

st.markdown(DARK_STYLE, unsafe_allow_html=True)

SAVE_FOLDER = Path("saved_protocols")
SAVE_FOLDER.mkdir(exist_ok=True)

SESSION_DEFAULTS = {
    "data_from_builder": None,
    "data_name": "",
    "load_from_builder": False,
    "saved_payload": None,
    "from_saved": False,
    "mode_selection": "30d",
    "calc_mode_selection": "MCAP",
    "tvl_factor_selection": 1.0,
    "analysis_range_picker": None,
    "show_range_buttons": True,
    "show_range_slider": True,
    "revenue_axis_log": False,
    "show_rev_ema7": False,
    "show_rev_ema14": False,
    "show_rev_ema30": False,
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def coerce_to_date(value):
    """Return a date object from various inputs or None when conversion fails."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().date()
    if isinstance(value, (float, int)) and pd.isna(value):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().date()


def clamp_date_range(preferred, min_bound, max_bound):
    """Clamp a preferred date span so it stays within the allowed bounds."""
    lower = coerce_to_date(min_bound)
    upper = coerce_to_date(max_bound)
    if lower is None or upper is None:
        return (lower, upper)
    if lower > upper:
        lower, upper = upper, lower

    start_pref = end_pref = None
    if isinstance(preferred, (tuple, list)) and preferred:
        start_pref = coerce_to_date(preferred[0])
        end_pref = coerce_to_date(preferred[1] if len(preferred) > 1 else preferred[0])
    else:
        start_pref = end_pref = coerce_to_date(preferred)

    if start_pref is None or end_pref is None:
        start_pref, end_pref = lower, upper
    if start_pref > end_pref:
        start_pref, end_pref = end_pref, start_pref

    start_pref = max(lower, start_pref)
    end_pref = min(upper, end_pref)

    if start_pref > end_pref:
        start_pref = end_pref = lower

    return (start_pref, end_pref)


def pick_col(df: pd.DataFrame, patterns):
    if df is None or df.empty:
        return None
    for pattern in patterns:
        for col in df.columns:
            if col.lower() == pattern.lower():
                return col
        regex = re.compile(pattern, flags=re.I)
        for col in df.columns:
            if regex.search(str(col)):
                return col
    return None


def parse_num(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value)
    text = text.replace("R$", "").replace("US$", "").replace(" ", "")
    text = text.replace(".", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-eE]", "", text)
    try:
        return float(text)
    except Exception:
        return np.nan


def compute_swpe(df: pd.DataFrame, mode="30d", calc_mode="MCAP", tvl_factor=1.0):
    if df is None or df.empty:
        st.warning("Nenhum dado encontrado para calcular o SWPE.")
        return pd.DataFrame()

    work = df.copy()

    col_date = pick_col(work, [r"^date$", "timestamp", "time"])
    col_rev = pick_col(work, [r"^daily\\s*revenue$", "revenue", "rev", "fee"])
    col_mcap = pick_col(work, [r"^mcap$", "market\\s*cap", "marketcap"])
    col_tvl = pick_col(work, [r"^tvl$", "total\\s*value\\s*locked"])

    missing = []
    if col_rev is None:
        missing.append("Daily Revenue")
    if col_mcap is None:
        missing.append("MCAP")
    if missing:
        st.error("Colunas obrigatorias ausentes: " + ", ".join(missing))
        return pd.DataFrame()

    work["rev_daily"] = work[col_rev].apply(parse_num)
    work["mcap"] = work[col_mcap].apply(parse_num)

    if col_tvl:
        work["tvl"] = work[col_tvl].apply(parse_num)
    else:
        work["tvl"] = np.nan

    if col_date:
        work["Date"] = pd.to_datetime(work[col_date], errors="coerce")
    else:
        work["Date"] = pd.to_datetime(work.index, errors="coerce")

    work = work.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if mode == "30d":
        work["revX"] = work["rev_daily"].rolling(30, min_periods=7).mean()
    elif mode == "7d":
        work["revX"] = work["rev_daily"].rolling(7, min_periods=3).mean()
    else:
        work["revX"] = work["rev_daily"]

    work["rev_annual"] = work["revX"] * 365
    work["rev_annual"] = work["rev_annual"].replace({0.0: np.nan})

    if calc_mode == "MCAP - TVL":
        if work["tvl"].isna().all():
            st.error("Modo 'MCAP - TVL' exige coluna de TVL.")
            return pd.DataFrame()
        work["num_used"] = work["mcap"] - (work["tvl"] * float(tvl_factor))
        work["num_used"] = work["num_used"].clip(lower=1e-9)
    else:
        work["num_used"] = work["mcap"].clip(lower=1e-9)

    work["SWPE"] = work["num_used"] / work["rev_annual"]
    work["SWPE"] = work["SWPE"].replace([np.inf, -np.inf], np.nan)

    work["EMA_7"] = work["rev_daily"].ewm(span=7, adjust=False).mean()
    work["EMA_14"] = work["rev_daily"].ewm(span=14, adjust=False).mean()
    work["EMA_30"] = work["rev_daily"].ewm(span=30, adjust=False).mean()

    return work


def plot_swpe(
    df: pd.DataFrame,
    title: str,
    mean_line=None,
    median_line=None,
    show_revenue_ema7: bool = False,
    show_revenue_ema14: bool = False,
    show_revenue_ema30: bool = False,
    revenue_axis_log: bool = False,
    show_range_buttons: bool = True,
    show_range_slider: bool = True,
) -> go.Figure:
    """Renderiza o grafico principal de SWPE com opcoes interativas."""
    if df.empty:
        return go.Figure()

    dates = pd.to_datetime(df["Date"])
    swpe_values = df["SWPE"]
    median_value = (
        float(median_line)
        if median_line is not None and not np.isnan(median_line)
        else float(swpe_values.median(skipna=True))
    )
    mean_value = (
        float(mean_line)
        if mean_line is not None and not np.isnan(mean_line)
        else None
    )
    revenue_daily = df.get("rev_daily")
    if revenue_daily is None:
        revenue_daily = pd.Series(np.nan, index=df.index)

    diff_from_median = swpe_values - median_value
    customdata = np.column_stack(
        [
            diff_from_median.fillna(np.nan).to_numpy(),
            revenue_daily.fillna(np.nan).to_numpy(),
        ]
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=swpe_values,
            mode="lines",
            name="SWPE",
            line=dict(color="#a855f7", width=2.6, shape="spline", smoothing=1.05),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%d/%m/%Y}</b><br>"
                "SWPE: %{y:.3f}<br>"
                "Distância à mediana: %{customdata[0]:+.3f}<br>"
                "Revenue diário: US$ %{customdata[1]:,.0f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    rolling_conf = [
        (7, "rgba(88, 171, 255, 0.55)", "Média 7D"),
        (30, "rgba(255, 146, 208, 0.45)", "Média 30D"),
    ]
    for span, color, label in rolling_conf:
        rolling_avg = swpe_values.rolling(span, min_periods=max(2, span // 2)).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_avg,
                mode="lines",
                name=f"SWPE {label}",
                line=dict(color=color, width=1.2, dash="dot"),
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    median_array = np.full(len(df), median_value)
    below_mask = swpe_values < median_value
    if np.any(below_mask):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.where(below_mask, median_array, np.nan),
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="median-zones",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.where(below_mask, swpe_values, np.nan),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(111, 227, 166, 0.28)",
                hoverinfo="skip",
                name="Barato (< mediana)",
                legendgroup="median-zones",
                legendrank=50,
            ),
            secondary_y=False,
        )

    above_mask = swpe_values >= median_value
    if np.any(above_mask):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.where(above_mask, median_array, np.nan),
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="median-zones",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.where(above_mask, swpe_values, np.nan),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 120, 162, 0.24)",
                hoverinfo="skip",
                name="Caro (> mediana)",
                legendgroup="median-zones",
                legendrank=51,
            ),
            secondary_y=False,
        )

    revenue_traces = [
        ("EMA_7", "#3ad6ff", "Revenue EMA-7", show_revenue_ema7),
        ("EMA_14", "#7dd87d", "Revenue EMA-14", show_revenue_ema14),
        ("EMA_30", "#a1ffce", "Revenue EMA-30", show_revenue_ema30),
    ]
    for column, color, label, enabled in revenue_traces:
        if enabled and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df[column],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=(
                        "<b>%{x|%d/%m/%Y}</b><br>"
                        f"{label}: US$ "+"%{y:,.0f}<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

    if mean_value is not None:
        fig.add_hline(
            y=mean_value,
            line=dict(color="rgba(231, 236, 255, 0.38)", width=1.6, dash="dash"),
            annotation_text=f"Média {mean_value:.3f}",
            annotation_position="top left",
            secondary_y=False,
        )
    if median_value is not None and not np.isnan(median_value):
        fig.add_hline(
            y=median_value,
            line=dict(color="rgba(255, 215, 141, 0.45)", width=1.6, dash="dot"),
            annotation_text=f"Mediana {median_value:.3f}",
            annotation_position="bottom left",
            secondary_y=False,
        )

    xaxis_kwargs = dict(showgrid=False)
    if show_range_buttons:
        xaxis_kwargs["rangeselector"] = dict(
            buttons=[
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(count=90, label="90D", step="day", stepmode="backward"),
                dict(count=180, label="180D", step="day", stepmode="backward"),
                dict(count=365, label="1Y", step="day", stepmode="backward"),
                dict(step="all", label="Tudo"),
            ],
            bgcolor="rgba(20, 24, 38, 0.9)",
            activecolor="rgba(91, 123, 255, 0.85)",
            font=dict(color="#e7ecff"),
        )
    if show_range_slider:
        xaxis_kwargs["rangeslider"] = dict(
            visible=True,
            bgcolor="rgba(20, 24, 38, 0.78)",
            bordercolor="rgba(86, 120, 255, 0.28)",
            thickness=0.08,
        )
    else:
        xaxis_kwargs["rangeslider"] = dict(visible=False)
    fig.update_xaxes(**xaxis_kwargs)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#0f1116",
        paper_bgcolor="#0f1116",
        font=dict(color="#e7ecff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(18, 22, 34, 0.9)",
            bordercolor="rgba(91, 123, 255, 0.6)",
            font=dict(color="#f0f3ff"),
        ),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(231, 236, 255, 0.08)",
        title_text="SWPE",
        secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=False,
        title_text="Revenue (EMA)",
        secondary_y=True,
        type="log" if revenue_axis_log else "linear",
    )

    return fig


def percent_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Calcula a variação percentual, retornando None quando não aplicável."""
    if current is None or previous is None:
        return None
    if isinstance(current, (float, int)) and isinstance(previous, (float, int)):
        if np.isnan(current) or np.isnan(previous) or previous == 0:
            return None
        return (float(current) / float(previous) - 1.0) * 100.0
    return None


def format_delta_badge(label: str, delta: Optional[float]) -> str:
    """Gera HTML para representar a variação."""
    if delta is None:
        return f"<span>{label}: s/d</span>"
    css_class = "positive" if delta >= 0 else "negative"
    return f'<span class="{css_class}">{label}: {delta:+.2f}%</span>'


def format_currency(value: Optional[float]) -> str:
    """Formata valores monetários com símbolo US$ e separador de milhar."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    formatted = f"{float(value):,.0f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"US$ {formatted}"


def format_swpe_value(value: Optional[float]) -> str:
    """Formata valores de SWPE com três casas decimais."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{float(value):.3f}"


def render_kpi_card(title: str, value_text: str, delta7: Optional[float], delta30: Optional[float]):
    """Renderiza um card de KPI com variações."""
    delta_html = "".join(
        [
            format_delta_badge("7d", delta7),
            format_delta_badge("30d", delta30),
        ]
    )
    st.markdown(
        f"""
        <div class="swpe-kpi-card">
            <div class="swpe-kpi-title">{title}</div>
            <div class="swpe-kpi-value">{value_text}</div>
            <div class="swpe-kpi-delta">{delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_reference(
    df: pd.DataFrame,
    column: str,
    cutoff: pd.Timestamp,
    reducer: Callable[[pd.Series], float],
) -> Optional[float]:
    """Obtém o valor agregado até uma data limite."""
    if column not in df.columns or df.empty:
        return None
    subset = df.loc[df["Date"] <= cutoff, column].dropna()
    if subset.empty:
        return None
    try:
        return float(reducer(subset))
    except Exception:
        return None


def parse_date_series(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        median_val = pd.Series(series.dropna()).median()
        unit = "ms" if median_val and median_val > 1e11 else "s"
        parsed = pd.to_datetime(series, unit=unit, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.tz_localize(None)


def normalize_numeric(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    comma_avg = text.str.count(",").mean()
    dot_avg = text.str.count(r"\.").mean()
    if comma_avg > dot_avg:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    text = text.str.replace(" ", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def dedupe_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    work = df[[date_col, value_col]].copy()
    work[date_col] = parse_date_series(work[date_col])
    work[value_col] = normalize_numeric(work[value_col])
    work = work.dropna(subset=[date_col])
    work = work.sort_values(date_col)
    return work.groupby(date_col, as_index=False)[value_col].last()


def transform_data(rev_df: pd.DataFrame, tvl_df: pd.DataFrame, mcap_df: pd.DataFrame, join_mode: str = "tvl_left"):
    r_date = pick_col(rev_df, [r"^date$", "time", "timestamp"]) or rev_df.columns[0]
    r_val = pick_col(rev_df, ["rev", "fee", "revenue"]) or rev_df.columns[-1]

    t_date = pick_col(tvl_df, [r"^date$", "time", "timestamp"]) or tvl_df.columns[0]
    t_val = pick_col(tvl_df, ["tvl"]) or tvl_df.columns[-1]

    m_date = pick_col(mcap_df, [r"^date$", "time", "timestamp"]) or mcap_df.columns[0]
    m_val = pick_col(mcap_df, ["mcap", "market.?cap", "token.?mcap", "float.?cap"]) or mcap_df.columns[-1]

    R = dedupe_daily(rev_df, r_date, r_val).rename(columns={r_date: "Date", r_val: "Daily Revenue"})
    T = dedupe_daily(tvl_df, t_date, t_val).rename(columns={t_date: "Date", t_val: "TVL"})
    M = dedupe_daily(mcap_df, m_date, m_val).rename(columns={m_date: "Date", m_val: "MCAP"})

    if join_mode == "inner":
        base = T.merge(M, on="Date", how="inner").merge(R, on="Date", how="inner")
    elif join_mode == "revenue_left":
        base = R.merge(T, on="Date", how="left").merge(M, on="Date", how="left")
    elif join_mode == "mcap_left":
        base = M.merge(T, on="Date", how="left").merge(R, on="Date", how="left")
    else:
        base = T.merge(M, on="Date", how="left").merge(R, on="Date", how="left")

    base["Annualized Revenue"] = (base["Daily Revenue"] * 365).round()
    base = base[["Date", "Daily Revenue", "Annualized Revenue", "MCAP", "TVL"]]
    drop_mask = base[["Daily Revenue", "MCAP"]].isna().all(axis=1)
    base = base[~drop_mask].copy()
    base = base.sort_values("Date", ascending=False).reset_index(drop=True)
    return base, R, T, M


def describe_range(label: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"**{label}**: vazio"
    dates = pd.to_datetime(df["Date"], errors="coerce")
    return f"**{label}** | linhas: {len(df)} | min: {dates.min().date()} | max: {dates.max().date()}"


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    content = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_excel(io.BytesIO(content))
    return df


def read_csv_upload(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(content), sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content))
    return df


def save_analysis(table_df: pd.DataFrame, metadata: dict) -> str:
    base_label = metadata.get("label") or "swpe"
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", base_label).strip("_") or "swpe"
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    identifier = f"{sanitized}_{stamp}"
    csv_path = SAVE_FOLDER / f"{identifier}.csv"
    meta_path = SAVE_FOLDER / f"{identifier}.json"

    metadata = metadata.copy()
    metadata.update({
        "id": identifier,
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    })

    table_df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return identifier


def list_saved_analyses():
    items = []
    for meta_file in SAVE_FOLDER.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        csv_path = SAVE_FOLDER / f"{meta.get('id', '')}.csv"
        if not csv_path.exists():
            continue
        meta["csv_path"] = str(csv_path)
        meta["meta_path"] = str(meta_file)
        items.append(meta)
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return items


def load_saved_analysis(identifier: str):
    meta_path = SAVE_FOLDER / f"{identifier}.json"
    csv_path = SAVE_FOLDER / f"{identifier}.csv"
    if not meta_path.exists() or not csv_path.exists():
        raise FileNotFoundError("Análise salva não encontrada.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c != "Date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df, metadata


def delete_saved_analysis(identifier: str):
    meta_path = SAVE_FOLDER / f"{identifier}.json"
    csv_path = SAVE_FOLDER / f"{identifier}.csv"
    if meta_path.exists():
        meta_path.unlink(missing_ok=True)
    if csv_path.exists():
        csv_path.unlink(missing_ok=True)



def render_dashboard():
    st.title("SWPE Control Center")
    st.caption("Centralize o cálculo de SWPE para diferentes protocolos, armazene históricos e compare períodos.")

    sticky_header = st.container()

    saved_records = list_saved_analyses()
    saved_placeholder = "-- selecionar --"
    saved_options = {saved_placeholder: None}
    for meta in saved_records:
        label = meta.get("display_label") or meta.get("label") or meta.get("id")
        created = meta.get("created_at", "")[:10]
        subtitle = meta.get("source_name") or "dataset"
        option = f"{label} | {subtitle} | {created}"
        saved_options[option] = meta["id"]

    st.sidebar.subheader("Biblioteca de análises")
    selected_saved = st.sidebar.selectbox("Abrir análise", list(saved_options.keys()), key="saved_selector")
    selected_id = saved_options[selected_saved]

    load_clicked = st.sidebar.button("Carregar análise", key="load_saved_btn")
    delete_clicked = st.sidebar.button("Excluir análise selecionada", key="delete_saved_btn")

    if load_clicked and selected_id:
        try:
            df_saved, meta = load_saved_analysis(selected_id)
            st.session_state["saved_payload"] = {"data": df_saved, "meta": meta}
            st.session_state["from_saved"] = True
            st.session_state["mode_selection"] = meta.get("mode", "30d")
            st.session_state["calc_mode_selection"] = meta.get("calc_mode", "MCAP")
            st.session_state["tvl_factor_selection"] = float(meta.get("tvl_factor", 1.0))
            preferred_start = coerce_to_date(meta.get("start_date"))
            preferred_end = coerce_to_date(meta.get("end_date"))
            if preferred_start and preferred_end:
                st.session_state["analysis_range_picker"] = (preferred_start, preferred_end)
            st.success(f"Análise {meta.get('display_label', meta.get('id'))} carregada.")
        except Exception as exc:
            st.error(f"Falha ao carregar análise: {exc}")

    if delete_clicked and selected_id:
        delete_saved_analysis(selected_id)
        st.sidebar.success("Análise removida. Atualize a página para atualizar a lista.")

    st.sidebar.markdown("---")

    calc_mode_options = ["MCAP", "MCAP - TVL"]
    calc_mode_default = st.session_state.get("calc_mode_selection", "MCAP")
    st.sidebar.subheader("Fórmula do SWPE")
    calc_mode = st.sidebar.radio(
        "Fórmula do SWPE",
        calc_mode_options,
        index=calc_mode_options.index(calc_mode_default),
        key="calc_mode_selection"
    )

    st.sidebar.subheader("Peso do TVL")
    tvl_factor = st.sidebar.slider(
        "Peso do TVL",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state.get("tvl_factor_selection", 1.0)),
        step=0.05,
        key="tvl_factor_selection"
    )

    st.sidebar.subheader("Origem dos dados")
    uploaded_file = st.sidebar.file_uploader("Upload (CSV ou XLSX)", type=["csv", "xlsx"], key="dashboard_upload")
    if st.sidebar.button("Limpar dados recebidos", key="clear_builder"):
        st.session_state["data_from_builder"] = None
        st.session_state["data_name"] = ""
        st.session_state["load_from_builder"] = False
        st.session_state["saved_payload"] = None
        st.session_state["from_saved"] = False
        st.session_state["analysis_range_picker"] = None
        st.sidebar.success("Dados em memoria foram limpos.")

    status_messages: list[Tuple[str, str]] = []
    df_swpe = None
    table_for_save = None
    source_name = ""
    saved_view = False
    analysis_start = analysis_end = None
    date_bounds: Optional[Tuple[date, date]] = None
    df_source = None

    mode = st.session_state.get("mode_selection", "30d")

    if st.session_state.get("from_saved") and st.session_state.get("saved_payload"):
        saved_view = True
        payload = st.session_state["saved_payload"]
        df_saved = payload["data"].copy()
        meta = payload["meta"]
        source_name = meta.get("source_name") or f"Análise {meta.get('id')}"
        date_series = pd.to_datetime(df_saved.get("Date"), errors="coerce")
        if date_series.notna().any():
            min_date = date_series.min().date()
            max_date = date_series.max().date()
            date_bounds = (min_date, max_date)
            preferred_range = clamp_date_range(
                st.session_state.get("analysis_range_picker"), min_date, max_date
            )
            if preferred_range is None:
                meta_range = (
                    coerce_to_date(meta.get("start_date")),
                    coerce_to_date(meta.get("end_date")),
                )

                preferred_range = clamp_date_range(meta_range, min_date, max_date)
            if preferred_range is None:
                preferred_range = (min_date, max_date)
            st.session_state["analysis_range_picker"] = preferred_range
            start_date = coerce_to_date(preferred_range[0])
            end_date = coerce_to_date(preferred_range[1])
            if start_date and end_date:
                if start_date > end_date:
                    start_date, end_date = end_date, start_date
                mask = date_series.between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
                filtered = df_saved.loc[mask].copy()
                if filtered.empty:
                    status_messages.append(("warning", "Nenhuma linha após aplicar o período selecionado."))
                else:
                    df_swpe = filtered
                    analysis_start = pd.Timestamp(start_date)
                    analysis_end = pd.Timestamp(end_date)
            else:
                status_messages.append(("info", "Selecione um intervalo válido na barra superior."))
        else:
            status_messages.append(("warning", "Datas indisponíveis para esta análise salva."))
            df_swpe = df_saved.copy()
            valid_saved = date_series.dropna()
            if not valid_saved.empty:
                analysis_start = valid_saved.min()
                analysis_end = valid_saved.max()
    else:
        st.session_state["from_saved"] = False
        st.session_state["saved_payload"] = None

        if uploaded_file is not None:
            df_source = read_uploaded_table(uploaded_file)
            source_name = uploaded_file.name
            st.sidebar.success(f"Arquivo carregado: {source_name}")
        elif st.session_state.get("load_from_builder") and st.session_state.get("data_from_builder") is not None:
            df_source = st.session_state["data_from_builder"].copy()
            source_name = st.session_state.get("data_name") or "Transformacao"
            st.sidebar.success("Dados recebidos da pagina Transformacao.")
            st.session_state["load_from_builder"] = False
        elif st.session_state.get("data_from_builder") is not None:
            df_source = st.session_state["data_from_builder"].copy()
            source_name = st.session_state.get("data_name") or "Transformacao"
            st.sidebar.info(f"Usando dados da Transformacao: {source_name}")

        if df_source is None or df_source.empty:
            status_messages.append(("info", "Carregue um arquivo na barra lateral ou gere dados pela aba Transformacao."))
        else:
            col_date = pick_col(df_source, [r"^date$", "timestamp", "time"])
            if not col_date:
                status_messages.append(("error", "Coluna de datas não encontrada no dataset."))
            else:
                date_series = pd.to_datetime(df_source[col_date], errors="coerce")
                valid_dates = date_series.dropna()
                if valid_dates.empty:
                    status_messages.append(("warning", "Nenhuma data válida encontrada no dataset."))
                else:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    date_bounds = (min_date, max_date)
                    preferred_range = clamp_date_range(
                        st.session_state.get("analysis_range_picker"), min_date, max_date
                    )
                    if preferred_range is None:
                        start_guess = max_date - timedelta(days=180)
                        if start_guess < min_date:
                            start_guess = min_date
                        preferred_range = (start_guess, max_date)
                    st.session_state["analysis_range_picker"] = preferred_range
                    start_date = coerce_to_date(preferred_range[0])
                    end_date = coerce_to_date(preferred_range[1])
                    if start_date and end_date:
                        if start_date > end_date:
                            start_date, end_date = end_date, start_date
                        start_ts = pd.Timestamp(start_date)
                        end_ts = pd.Timestamp(end_date)
                        mask = date_series.between(start_ts, end_ts, inclusive="both")
                        filtered_source = df_source.loc[mask].copy()
                        if filtered_source.empty:
                            status_messages.append(("warning", "Nenhuma linha após aplicar o período selecionado."))
                        else:
                            analysis_start = start_ts
                            analysis_end = end_ts
                            df_swpe = compute_swpe(filtered_source, mode=mode, calc_mode=calc_mode, tvl_factor=tvl_factor)
                            if df_swpe.empty:
                                status_messages.append(("warning", "Não foi possível calcular o SWPE com o dataset atual."))
                            else:
                                df_swpe = df_swpe.dropna(subset=["Date"]).reset_index(drop=True)
                    else:
                        status_messages.append(("info", "Selecione um intervalo válido na barra superior."))

    current_range = st.session_state.get("analysis_range_picker")
    if isinstance(current_range, tuple):
        current_range = (coerce_to_date(current_range[0]), coerce_to_date(current_range[1]))
    elif current_range:
        coerced = coerce_to_date(current_range)
        current_range = (coerced, coerced)
    else:
        current_range = None

    with sticky_header:
        st.markdown('<div class="swpe-sticky-marker"></div>', unsafe_allow_html=True)
        col_interval, col_window, col_source = st.columns([2.6, 1.4, 1.2])
        with col_interval:
            if date_bounds:
                min_date, max_date = date_bounds
                start_value, end_value = current_range or (min_date, max_date)
                if start_value is None or end_value is None:
                    start_value, end_value = min_date, max_date
                st.date_input(
                    "Intervalo considerado",
                    value=(start_value, end_value),
                    min_value=min_date,
                    max_value=max_date,
                    key="analysis_range_picker",
                    format="DD/MM/YYYY"
                )
            else:
                st.caption("Intervalo considerado: carregue dados para configurar.")
        with col_window:
            st.radio(
                "Janela da receita",
                ["30d", "7d", "daily"],
                index=["30d", "7d", "daily"].index(mode),
                key="mode_selection"
            )
        with col_source:
            if source_name:
                st.markdown(f"<small>Fonte atual</small><br><strong>{source_name}</strong>", unsafe_allow_html=True)
            elif saved_view:
                st.markdown("<small>Fonte atual</small><br><strong>Análise salva</strong>", unsafe_allow_html=True)
            else:
                st.caption("Selecione uma fonte na barra lateral.")

    for level, message in status_messages:
        getattr(st, level)(message)

    if df_swpe is None or df_swpe.empty:
        return

    date_series = pd.to_datetime(df_swpe["Date"], errors="coerce")
    valid_dates = date_series.dropna()
    if analysis_start is None and not valid_dates.empty:
        analysis_start = valid_dates.min()
    if analysis_end is None and not valid_dates.empty:
        analysis_end = valid_dates.max()

    if source_name:
        st.markdown(f"**Fonte utilizada:** `{source_name}`")
    elif saved_view:
        st.markdown("**Fonte utilizada:** `Análise salva`")

    if analysis_start is not None and analysis_end is not None:
        st.caption(f"Período analisado: {analysis_start.strftime('%d/%m/%Y')} → {analysis_end.strftime('%d/%m/%Y')}")

    clean_swpe = df_swpe["SWPE"].dropna()
    latest_swpe = clean_swpe.iloc[-1] if not clean_swpe.empty else np.nan
    mean_swpe = clean_swpe.mean() if not clean_swpe.empty else np.nan
    median_swpe = clean_swpe.median() if not clean_swpe.empty else np.nan

    ema7_series = df_swpe["EMA_7"].dropna() if "EMA_7" in df_swpe else pd.Series(dtype=float)
    ema30_series = df_swpe["EMA_30"].dropna() if "EMA_30" in df_swpe else pd.Series(dtype=float)
    ema7_latest = float(ema7_series.iloc[-1]) if not ema7_series.empty else None
    ema30_latest = float(ema30_series.iloc[-1]) if not ema30_series.empty else None

    last_timestamp = valid_dates.max() if not valid_dates.empty else None

    delta_current_7 = delta_current_30 = None
    delta_mean_7 = delta_mean_30 = None
    delta_median_7 = delta_median_30 = None
    if last_timestamp is not None:
        ref_7 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=7), lambda s: s.iloc[-1])
        ref_30 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=30), lambda s: s.iloc[-1])
        delta_current_7 = percent_change(latest_swpe, ref_7)
        delta_current_30 = percent_change(latest_swpe, ref_30)

        ref_mean_7 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=7), lambda s: s.mean())
        ref_mean_30 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=30), lambda s: s.mean())
        delta_mean_7 = percent_change(mean_swpe, ref_mean_7)
        delta_mean_30 = percent_change(mean_swpe, ref_mean_30)

        ref_median_7 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=7), lambda s: s.median())
        ref_median_30 = metric_reference(df_swpe, "SWPE", last_timestamp - timedelta(days=30), lambda s: s.median())
        delta_median_7 = percent_change(median_swpe, ref_median_7)
        delta_median_30 = percent_change(median_swpe, ref_median_30)

    tabs = st.tabs(["Visão Geral", "SWPE", "Tabela", "Análises"])

    with tabs[0]:
        st.subheader("Resumo geral")
        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            render_kpi_card("SWPE atual", format_swpe_value(latest_swpe), delta_current_7, delta_current_30)
        with kpi_cols[1]:
            render_kpi_card("SWPE média", format_swpe_value(mean_swpe), delta_mean_7, delta_mean_30)
        with kpi_cols[2]:
            render_kpi_card("SWPE mediana", format_swpe_value(median_swpe), delta_median_7, delta_median_30)

        rev_cols = st.columns(2)
        with rev_cols[0]:
            render_kpi_card("Revenue EMA-7", format_currency(ema7_latest), None, None)
        with rev_cols[1]:
            render_kpi_card("Revenue EMA-30", format_currency(ema30_latest), None, None)

        if analysis_start is not None and analysis_end is not None:
            st.caption(f"Período considerado: {analysis_start.strftime('%d/%m/%Y')} → {analysis_end.strftime('%d/%m/%Y')}")

    with tabs[1]:
        st.subheader("Curva SWPE")
        controls = st.container()
        with controls:
            st.markdown('<div class="swpe-chart-marker"></div>', unsafe_allow_html=True)
            toggles_row = st.columns(3)
            with toggles_row[0]:
                st.checkbox("Botões de range", key="show_range_buttons")
                st.checkbox("Range slider", key="show_range_slider")
            with toggles_row[1]:
                st.checkbox("Eixo revenue log", key="revenue_axis_log")
            with toggles_row[2]:
                st.caption("EMAs de revenue")
                st.checkbox("EMA-7", key="show_rev_ema7")
                st.checkbox("EMA-14", key="show_rev_ema14")
                st.checkbox("EMA-30", key="show_rev_ema30")

        subtitle = f"Modo {mode.upper()} | {calc_mode}"
        if calc_mode == "MCAP - TVL":
            subtitle += f" | TVL x{float(tvl_factor):.2f}"
        fig = plot_swpe(
            df_swpe,
            f"SWPE - {subtitle}",
            mean_line=mean_swpe,
            median_line=median_swpe,
            show_revenue_ema7=st.session_state.get("show_rev_ema7", False),
            show_revenue_ema14=st.session_state.get("show_rev_ema14", False),
            show_revenue_ema30=st.session_state.get("show_rev_ema30", False),
            revenue_axis_log=st.session_state.get("revenue_axis_log", False),
            show_range_buttons=st.session_state.get("show_range_buttons", True),
            show_range_slider=st.session_state.get("show_range_slider", True),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Faixas sombreadas destacam períodos em que o SWPE está abaixo (barato) ou acima (caro) da mediana.")

    display_cols = ["Date", "rev_daily", "rev_annual", "mcap", "tvl", "num_used", "SWPE", "EMA_7", "EMA_14", "EMA_30"]
    available_cols = [c for c in display_cols if c in df_swpe.columns]
    table_for_save = df_swpe[available_cols].copy()
    table_df = table_for_save.copy()
    if "Date" in table_df.columns:
        table_df["Date"] = pd.to_datetime(table_df["Date"], errors="coerce").dt.strftime("%d/%m/%Y")

    with tabs[2]:
        st.subheader("Tabela detalhada")
        st.dataframe(table_df, use_container_width=True)
        csv_bytes = table_for_save.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar CSV com SWPE",
            data=csv_bytes,
            file_name="swpe_resultado.csv",
            mime="text/csv"
        )

    with tabs[3]:
        st.subheader("Análises complementares")
        if clean_swpe.empty:
            st.info("Sem dados suficientes para comparativos adicionais.")
        else:
            low = df_swpe[["Date", "SWPE"]].dropna().nsmallest(5, "SWPE").copy()
            high = df_swpe[["Date", "SWPE"]].dropna().nlargest(5, "SWPE").copy()
            low["Date"] = pd.to_datetime(low["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
            high["Date"] = pd.to_datetime(high["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
            col_low, col_high = st.columns(2)
            with col_low:
                st.markdown("**Top 5 períodos baratos**")
                st.dataframe(low.rename(columns={"Date": "Data", "SWPE": "SWPE"}), hide_index=True, use_container_width=True)
            with col_high:
                st.markdown("**Top 5 períodos caros**")
                st.dataframe(high.rename(columns={"Date": "Data", "SWPE": "SWPE"}), hide_index=True, use_container_width=True)

    if not saved_view:
        st.sidebar.subheader("Salvar análise atual")
        default_name = re.sub(r"[^A-Za-z0-9_-]+", "_", (source_name or "analise").split('.')[0]) or "analise"
        protocol_name = st.sidebar.text_input("Identificador", value=default_name, key="save_name")
        notes = st.sidebar.text_input("Tag opcional", value="", key="save_tag")
        if st.sidebar.button("Salvar resultado", key="save_button"):
            metadata = {
                "label": protocol_name,
                "display_label": protocol_name,
                "tag": notes,
                "mode": mode,
                "calc_mode": calc_mode,
                "tvl_factor": float(tvl_factor),
                "source_name": source_name,
                "start_date": analysis_start.isoformat() if analysis_start is not None else None,
                "end_date": analysis_end.isoformat() if analysis_end is not None else None,
            }
            try:
                identifier = save_analysis(table_for_save, metadata)
                st.sidebar.success(f"Análise salva como {identifier}")
            except Exception as exc:
                st.sidebar.error(f"Erro ao salvar arquivo: {exc}")
    else:
        st.sidebar.info("Análises carregadas não podem ser sobrescritas diretamente. Gere nova rodada para salvar novamente.")

def render_transform():
    st.title("Transformacao (3 CSV -> SWPE)")
    st.caption("Normalize seus arquivos de Revenue, TVL e MCAP e envie direto para o dashboard.")

    left, right = st.columns([1, 1])
    with left:
        rev_file = st.file_uploader("Revenue CSV", type=["csv"], key="rev_file")
        tvl_file = st.file_uploader("TVL CSV", type=["csv"], key="tvl_file")
        mcap_file = st.file_uploader("MCAP CSV", type=["csv"], key="mcap_file")
    with right:
        join_mode = st.selectbox(
            "Base do merge",
            options=["tvl_left", "revenue_left", "mcap_left", "inner"],
            index=0,
            help="Escolha qual base deve prevalecer quando houver datas faltantes."
        )
        st.markdown("""
        **Dicas**
        - Use a mesma janela temporal para os 3 arquivos.
        - Depois de gerar, envie para o Dashboard para concluir o calculo do SWPE.
        """)

    st.markdown("---")

    if not (rev_file and tvl_file and mcap_file):
        st.info("Anexe os **3 arquivos CSV** para liberar os botoes de processamento.")
        return

    try:
        rev_df = read_csv_upload(rev_file)
        tvl_df = read_csv_upload(tvl_file)
        mcap_df = read_csv_upload(mcap_file)
    except Exception as exc:
        st.error(f"Erro ao ler os arquivos: {exc}")
        return

    try:
        final_df, R, T, M = transform_data(rev_df, tvl_df, mcap_df, join_mode=join_mode)
    except Exception as exc:
        st.error(f"Falha durante o tratamento dos dados: {exc}")
        return

    if final_df.empty:
        st.warning("O resultado ficou vazio apos o merge. Verifique os arquivos de origem e o modo selecionado.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas (final)", len(final_df))
    c2.metric("Data mais recente", str(pd.to_datetime(final_df["Date"]).max().date()))
    c3.metric("Data mais antiga", str(pd.to_datetime(final_df["Date"]).min().date()))
    c4.metric("Modo de merge", join_mode)

    st.caption(describe_range("Revenue (1/dia)", R))
    st.caption(describe_range("TVL (1/dia)", T))
    st.caption(describe_range("MCAP (1/dia)", M))

    st.subheader("Previa do dataset tratado")
    st.dataframe(final_df.head(50), width="stretch")

    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="SWPE")
    except Exception:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="SWPE")
    excel_bytes = excel_buffer.getvalue()

    col_download, col_send = st.columns(2)
    with col_download:
        st.download_button(
            "Baixar XLSX tratado",
            data=excel_bytes,
            file_name="SWPE_tratado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch"
        )

    with col_send:
        if st.button("Enviar dados para o Dashboard", width="stretch"):
            st.session_state["data_from_builder"] = final_df.copy()
            st.session_state["data_name"] = f"Tratado_{join_mode}"
            st.session_state["load_from_builder"] = True
            st.session_state["saved_payload"] = None
            st.session_state["from_saved"] = False

            dates = pd.to_datetime(final_df["Date"], errors="coerce").dropna()
            if not dates.empty:
                max_date = dates.max().date()
                min_date = dates.min().date()
                start_last_year = max_date - timedelta(days=365)
                if start_last_year < min_date:
                    start_last_year = min_date
                st.session_state["analysis_range_picker"] = (start_last_year, max_date)
            else:
                st.session_state["analysis_range_picker"] = None

            st.success("Dados enviados! Abra a aba Dashboard para visualizar o SWPE.")


st.sidebar.title("SWPE Suite")
st.sidebar.write("Selecione a pagina para trabalhar.")
page = st.sidebar.radio("Menu", ["Dashboard", "Transformacao"], index=0)

if page == "Dashboard":
    render_dashboard()
else:
    render_transform()







