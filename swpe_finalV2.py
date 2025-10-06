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


def plot_swpe(df: pd.DataFrame, title: str, mean_line=None, median_line=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SWPE"],
            mode="lines",
            name="SWPE",
            line=dict(color="#a855f7", width=2.6, shape="spline", smoothing=1.1)
        ),
        secondary_y=False
    )

    for span, color, label in [
        (7, "rgba(88, 171, 255, 0.55)", "EMA-7"),
        (30, "rgba(255, 146, 208, 0.45)", "EMA-30"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SWPE"].rolling(span, min_periods=max(2, span // 2)).mean(),
                mode="lines",
                name=f"SWPE {label}",
                line=dict(color=color, width=1.2, dash="dot")
            ),
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA_7"],
            mode="lines",
            name="Revenue EMA-7",
            line=dict(color="#3ad6ff", width=1.5)
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA_14"],
            mode="lines",
            name="Revenue EMA-14",
            line=dict(color="#7dd87d", width=1.4)
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA_30"],
            mode="lines",
            name="Revenue EMA-30",
            line=dict(color="#a1ffce", width=1.2)
        ),
        secondary_y=True
    )

    if mean_line is not None and not np.isnan(mean_line):
        fig.add_hline(
            y=mean_line,
            line=dict(color="rgba(231, 236, 255, 0.38)", width=1.6, dash="dash"),
            annotation_text=f"Media {mean_line:.3f}",
            annotation_position="top left",
            secondary_y=False
        )
    if median_line is not None and not np.isnan(median_line):
        fig.add_hline(
            y=median_line,
            line=dict(color="rgba(255, 215, 141, 0.45)", width=1.6, dash="dot"),
            annotation_text=f"Mediana {median_line:.3f}",
            annotation_position="bottom left",
            secondary_y=False
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#0f1116",
        paper_bgcolor="#0f1116",
        font=dict(color="#e7ecff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(231, 236, 255, 0.08)", title_text="SWPE", secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text="Revenue (EMA)", secondary_y=True)

    return fig


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
        raise FileNotFoundError("Analise salva nao encontrada.")
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
    st.caption("Centralize o calculo de SWPE para diferentes protocolos, armazene historicos e compare periodos.")

    saved_records = list_saved_analyses()
    saved_placeholder = "-- selecionar --"
    saved_options = {saved_placeholder: None}
    for meta in saved_records:
        label = meta.get("display_label") or meta.get("label") or meta.get("id")
        created = meta.get("created_at", "")[:10]
        subtitle = meta.get("source_name") or "dataset"
        option = f"{label} | {subtitle} | {created}"
        saved_options[option] = meta["id"]

    interval_placeholder = st.sidebar.empty()
    with interval_placeholder:
        st.caption("Intervalo considerado: carregue dados para configurar.")

    st.sidebar.subheader("Janela da receita")
    mode = st.sidebar.radio(
        "Janela da receita",
        ["30d", "7d", "daily"],
        index=["30d", "7d", "daily"].index(st.session_state["mode_selection"]),
        key="mode_selection"
    )

    st.sidebar.subheader("Formula do SWPE")
    calc_mode = st.sidebar.radio(
        "Formula do SWPE",
        ["MCAP", "MCAP - TVL"],
        index=["MCAP", "MCAP - TVL"].index(st.session_state["calc_mode_selection"]),
        key="calc_mode_selection"
    )

    st.sidebar.subheader("Peso do TVL")
    tvl_factor = st.sidebar.slider(
        "Peso do TVL",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state["tvl_factor_selection"]),
        step=0.05,
        key="tvl_factor_selection"
    )

    st.sidebar.subheader("Biblioteca de analises")
    selected_saved = st.sidebar.selectbox("Abrir analise", list(saved_options.keys()), key="saved_selector")
    selected_id = saved_options[selected_saved]

    load_clicked = st.sidebar.button("Carregar analise", key="load_saved_btn")
    delete_clicked = st.sidebar.button("Excluir analise selecionada", key="delete_saved_btn")

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
            # Corrigido: só seta se ainda não existe
            if preferred_start and preferred_end and "analysis_range_picker" not in st.session_state:
                st.session_state["analysis_range_picker"] = (preferred_start, preferred_end)
                with interval_placeholder:
                    st.caption("Intervalo considerado carregado da analise salva.")
            else:
                if "analysis_range_picker" not in st.session_state:
                    st.session_state["analysis_range_picker"] = None
            st.success(f"Analise {meta.get('display_label', meta.get('id'))} carregada.")
        except Exception as exc:
            st.error(f"Falha ao carregar analise: {exc}")

    if delete_clicked and selected_id:
        delete_saved_analysis(selected_id)
        st.sidebar.success("Analise removida. Atualize a pagina para atualizar a lista.")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Origem dos dados")
    uploaded_file = st.sidebar.file_uploader("Upload (CSV ou XLSX)", type=["csv", "xlsx"], key="dashboard_upload")
    if st.sidebar.button("Limpar dados recebidos", key="clear_builder"):
        st.session_state["data_from_builder"] = None
        st.session_state["data_name"] = ""
        st.session_state["load_from_builder"] = False
        st.session_state["saved_payload"] = None
        st.session_state["from_saved"] = False
        if "analysis_range_picker" in st.session_state:
            del st.session_state["analysis_range_picker"]
        with interval_placeholder:
            st.caption("Intervalo considerado: carregue dados para configurar.")
        st.sidebar.success("Dados em memoria foram limpos.")

    df_swpe = None
    df_source = None
    source_name = ""
    analysis_start = analysis_end = None
    saved_view = False

    if st.session_state.get("from_saved") and st.session_state.get("saved_payload"):
        saved_view = True
        payload = st.session_state["saved_payload"]
        df_saved = payload["data"].copy()
        meta = payload["meta"]
        source_name = meta.get("source_name") or f"Analise {meta.get('id')}"
        mode = meta.get("mode", mode)
        calc_mode = meta.get("calc_mode", calc_mode)
        tvl_factor = float(meta.get("tvl_factor", tvl_factor))

        date_series = pd.to_datetime(df_saved.get("Date"), errors="coerce")
        if date_series.notna().any():
            min_date = date_series.min().date()
            max_date = date_series.max().date()
            preferred = st.session_state.get("analysis_range_picker")
            if preferred is None:
                preferred = (
                    coerce_to_date(meta.get("start_date")),
                    coerce_to_date(meta.get("end_date"))
                )
                # Corrigido: só seta se ainda não existe
                if preferred[0] and preferred[1] and "analysis_range_picker" not in st.session_state:
                    st.session_state["analysis_range_picker"] = preferred
            default_range = clamp_date_range(st.session_state.get("analysis_range_picker"), min_date, max_date)
            interval_placeholder.empty()
            with interval_placeholder:
                selected_range = st.date_input(
                    "Intervalo considerado",
                    value=default_range,
                    min_value=min_date,
                    max_value=max_date,
                    key="analysis_range_picker"
                )
            if isinstance(selected_range, tuple):
                start_date = coerce_to_date(selected_range[0])
                end_date = coerce_to_date(selected_range[1])
            else:
                start_date = end_date = coerce_to_date(selected_range)
            if start_date and end_date:
                if start_date > end_date:
                    start_date, end_date = end_date, start_date
                # Não setar session_state aqui!
                mask = date_series.between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
                filtered = df_saved.loc[mask].copy()
                if filtered.empty:
                    st.warning("Nenhuma linha apos aplicar o periodo selecionado.")
                    df_swpe = filtered
                else:
                    df_swpe = filtered
                    analysis_start = pd.Timestamp(start_date)
                    analysis_end = pd.Timestamp(end_date)
            else:
                df_swpe = df_saved.copy()
        else:
            interval_placeholder.empty()
            with interval_placeholder:
                st.caption("Datas indisponiveis para filtro nesta analise salva.")
            df_swpe = df_saved.copy()

        if analysis_start is None and meta.get("start_date"):
            try:
                analysis_start = pd.to_datetime(meta.get("start_date"))
            except Exception:
                analysis_start = None
        if analysis_end is None and meta.get("end_date"):
            try:
                analysis_end = pd.to_datetime(meta.get("end_date"))
            except Exception:
                analysis_end = None
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
            with interval_placeholder:
                st.caption("Intervalo considerado: carregue dados para configurar.")
            st.info("Carregue um arquivo no sidebar ou gere dados pela aba Transformacao.")
            return

        col_date = pick_col(df_source, [r"^date$", "timestamp", "time"])
        if not col_date:
            interval_placeholder.empty()
            with interval_placeholder:
                st.caption("Coluna de datas nao encontrada no dataset.")
            st.error("Coluna de datas nao encontrada no dataset.")
            return

        date_series = pd.to_datetime(df_source[col_date], errors="coerce")
        valid_dates = date_series.dropna()
        if valid_dates.empty:
            interval_placeholder.empty()
            with interval_placeholder:
                st.caption("Nao foi possivel converter as datas do arquivo.")
            st.error("Nao foi possivel converter as datas do arquivo.")
            return

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        preferred = st.session_state.get("analysis_range_picker") or (min_date, max_date)
        # Corrigido: só seta se ainda não existe
        if "analysis_range_picker" not in st.session_state:
            st.session_state["analysis_range_picker"] = clamp_date_range(preferred, min_date, max_date)
        default_range = clamp_date_range(st.session_state.get("analysis_range_picker"), min_date, max_date)
        interval_placeholder.empty()
        with interval_placeholder:
            selected_range = st.date_input(
                "Intervalo considerado",
                value=default_range,
                min_value=min_date,
                max_value=max_date,
                key="analysis_range_picker"
            )
        if isinstance(selected_range, tuple):
            start_date = coerce_to_date(selected_range[0])
            end_date = coerce_to_date(selected_range[1])
        else:
            start_date = end_date = coerce_to_date(selected_range)
        if not (start_date and end_date):
            st.warning("Defina um intervalo valido para continuar.")
            return
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        # Não setar session_state aqui!
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        mask = date_series.between(start_ts, end_ts, inclusive="both")
        df_filtered = df_source.loc[mask].copy()
        if df_filtered.empty:
            st.warning("Nenhuma linha apos aplicar o periodo selecionado.")
            return
        df_source = df_filtered
        analysis_start = start_ts
        analysis_end = end_ts

        df_swpe = compute_swpe(df_source, mode=mode, calc_mode=calc_mode, tvl_factor=tvl_factor)
        if df_swpe.empty:
            st.warning("Nao foi possivel calcular o SWPE com o dataset atual.")
            return
        df_swpe = df_swpe.dropna(subset=["Date"]).reset_index(drop=True)

    if df_swpe is None or df_swpe.empty:
        st.info("Nenhum dado pronto para exibicao.")
        return

    st.markdown(f"**Fonte utilizada:** `{source_name}`")
    if analysis_start is not None and analysis_end is not None:
        st.caption(f"Periodo analisado: {analysis_start.strftime('%d/%m/%Y')} -> {analysis_end.strftime('%d/%m/%Y')}")
    elif saved_view:
        st.caption("Periodo analisado: conforme analise salva")

    clean_swpe = df_swpe["SWPE"].dropna()
    latest_swpe = clean_swpe.iloc[-1] if not clean_swpe.empty else np.nan
    mean_swpe = clean_swpe.mean() if not clean_swpe.empty else np.nan
    median_swpe = clean_swpe.median() if not clean_swpe.empty else np.nan

    ema7 = df_swpe["EMA_7"].dropna()
    ema30 = df_swpe["EMA_30"].dropna()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SWPE atual", f"{latest_swpe:.3f}" if not np.isnan(latest_swpe) else "-")
    c2.metric("SWPE medio", f"{mean_swpe:.3f}" if not np.isnan(mean_swpe) else "-")
    c3.metric("SWPE mediano", f"{median_swpe:.3f}" if not np.isnan(median_swpe) else "-")
    c4.metric("Revenue EMA-7", f"{ema7.iloc[-1]:,.0f}" if not ema7.empty else "-")
    c5.metric("Revenue EMA-30", f"{ema30.iloc[-1]:,.0f}" if not ema30.empty else "-")

    subtitle = f"Modo {mode.upper()} | {calc_mode}"
    if calc_mode == "MCAP - TVL":
        subtitle += f" | TVL x{float(tvl_factor):.2f}"
    st.plotly_chart(
        plot_swpe(df_swpe, f"SWPE - {subtitle}", mean_line=mean_swpe, median_line=median_swpe),
        width="stretch"
    )

    display_cols = ["Date", "rev_daily", "rev_annual", "mcap", "tvl", "num_used", "SWPE", "EMA_7", "EMA_14", "EMA_30"]
    available_cols = [c for c in display_cols if c in df_swpe.columns]
    table_df = df_swpe[available_cols].copy()
    table_for_save = table_df.copy()
    if "Date" in table_df.columns:
        table_df["Date"] = pd.to_datetime(table_df["Date"], errors="coerce").dt.date

    st.subheader("Tabela detalhada")
    st.dataframe(table_df, width="stretch")

    csv_bytes = table_for_save.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar CSV com SWPE",
        data=csv_bytes,
        file_name="swpe_resultado.csv",
        mime="text/csv"
    )

    if not saved_view:
        st.sidebar.subheader("Salvar analise atual")
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
                st.sidebar.success(f"Analise salva como {identifier}")
            except Exception as exc:
                st.sidebar.error(f"Erro ao salvar arquivo: {exc}")
    else:
        st.sidebar.info("Analises carregadas nao podem ser sobrescritas diretamente. Gere nova rodada para salvar novamente.")

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







