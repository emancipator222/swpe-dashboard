import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re, os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SWPE DillaStyle V1.4", layout="wide")

# Estilos
st.markdown(
    """
    <style>
        .main { background-color: #e6f0ff; }
        section[data-testid="stSidebar"] { background-color: #f0f0f0; color: black; }
        section[data-testid="stSidebar"] * { color: black !important; }
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stFileUploader,
        section[data-testid="stSidebar"] .stButton {
            background-color: #d9d9d9; border-radius: 8px; padding: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 SWPE Dashboard — DillaStyle V1.4 (MCAP vs TVL)")

SAVE_FOLDER = "saved_protocols"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ---------------- HELPERS ----------------
def parse_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).strip()
    # normaliza números brasileiros (pontos de milhar e vírgulas decimais)
    s = s.replace("R$", "").replace("US$", "").replace(" ", "")
    # primeiro remove separadores de milhar '.' e depois troca ',' decimal por '.'
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9.\-eE]", "", s)
    try:
        return float(s)
    except:
        return np.nan

def pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for pat in candidates:
        # match exato
        if pat.lower() in cols: 
            return cols[pat.lower()]
        # regex
        regex = re.compile(pat, flags=re.I)
        for c in df.columns:
            if regex.search(c):
                return c
    return None

def compute_swpe(df, mode="30d", calc_mode="MCAP", tvl_factor=1.0):
    """
    calc_mode: "MCAP" ou "MCAP - TVL"
    tvl_factor: multiplicador do TVL quando calc_mode = "MCAP - TVL"
    """
    df = df.copy()

    # Detecta colunas
    col_date = pick_col(df, ["^date$", "timestamp", "time"])
    col_rev  = pick_col(df, ["^daily\\s*revenue$", "rev(enue)?", "fees?"])
    col_mcap = pick_col(df, ["^mcap$", "market\\s*cap", "marketcap"])
    col_tvl  = pick_col(df, ["^tvl$", "total\\s*value\\s*locked"])

    # Checagens mínimas
    missing = []
    if not col_rev:  missing.append("Daily Revenue")
    if not col_mcap: missing.append("MCAP")
    if missing:
        st.error(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
        return pd.DataFrame()

    # Parse e limpeza
    df["rev_daily"] = df[col_rev].apply(parse_num)
    df["mcap"] = df[col_mcap].apply(parse_num)

    if col_tvl:
        df["tvl"] = df[col_tvl].apply(parse_num)
    else:
        df["tvl"] = np.nan  # pode faltar se o usuário escolher modo MCAP

    # Janela da receita
    if mode == "30d":
        df["revX"] = df["rev_daily"].rolling(30, min_periods=7).mean()
    elif mode == "7d":
        df["revX"] = df["rev_daily"].rolling(7, min_periods=3).mean()
    else:
        df["revX"] = df["rev_daily"]

    # Anualização
    df["rev_annual"] = df["revX"] * 365

    # Numerador conforme modo
    if calc_mode == "MCAP - TVL":
        if df["tvl"].isna().all():
            st.error("Modo 'MCAP - TVL' selecionado, mas a coluna TVL não foi encontrada na planilha.")
            return pd.DataFrame()
        num = df["mcap"] - (df["tvl"] * float(tvl_factor))
        # Evita valores negativos/zero (eixo log/escala)
        df["num_used"] = np.maximum(num, 1e-9)
    else:
        df["num_used"] = df["mcap"]

    # SWPE
    df["SWPE"] = df["num_used"] / df["rev_annual"]

    # EMAs da receita (eixo secundário)
    df["EMA_7"]  = df["rev_daily"].ewm(span=7,  adjust=False).mean()
    df["EMA_14"] = df["rev_daily"].ewm(span=14, adjust=False).mean()
    df["EMA_30"] = df["rev_daily"].ewm(span=30, adjust=False).mean()

    # Datas
    if col_date:
        try:
            df["Date"] = pd.to_datetime(df[col_date])
        except Exception:
            df["Date"] = df.index
    else:
        df["Date"] = df.index

    # Ordena por data (se houver)
    df = df.sort_values("Date")

    # Limpa inf/NaN relevantes
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["SWPE", "rev_annual"])

    # Guarda metadados úteis
    df.attrs["calc_mode"]  = calc_mode
    df.attrs["tvl_factor"] = tvl_factor
    return df

def plot_swpe(df, title):
    median_swpe = df["SWPE"].median()
    min_swpe = df["SWPE"].min()
    mid_line = (min_swpe + median_swpe) / 2

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # SWPE (magenta)
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["SWPE"], mode="lines",
            name="SWPE",
            line=dict(color="#ff2db2", width=3, shape="spline", smoothing=1.3)
        ),
        secondary_y=False
    )

    # Linhas de referência
    fig.add_hline(
        y=median_swpe,
        line=dict(color="rgba(0,100,0,0.85)", dash="dash"),
        annotation_text=f"Median {median_swpe:.2f}",
        annotation_font=dict(color="rgba(0,100,0,1)")
    )
    fig.add_hline(
        y=mid_line,
        line=dict(color="rgba(0,100,0,0.7)", dash="dash"),
        annotation_text=f"Mid {mid_line:.2f}",
        annotation_font=dict(color="rgba(0,100,0,1)")
    )

    # Revenue EMAs (eixo secundário)
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["EMA_7"], mode="lines",
            name="Revenue EMA-7",
            line=dict(color="rgba(0,208,255,0.6)", width=1.5, shape="spline", smoothing=1.3)
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["EMA_14"], mode="lines",
            name="Revenue EMA-14",
            line=dict(color="rgba(60,179,113,0.6)", width=1.5, shape="spline", smoothing=1.3)
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["EMA_30"], mode="lines",
            name="Revenue EMA-30",
            line=dict(color="rgba(0,255,180,0.45)", width=1.2, shape="spline", smoothing=1.3)
        ),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="SWPE"),
        yaxis2=dict(showgrid=False, title="Revenue (EMA)"),
        plot_bgcolor="#e6f0ff",
        paper_bgcolor="#e6f0ff"
    )
    return fig

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configurações")

mode = st.sidebar.radio("Modo de cálculo da receita:", ["30d", "7d", "daily"], index=0)
calc_mode = st.sidebar.radio("Cálculo do SWPE:", ["MCAP", "MCAP - TVL"], index=0)

with st.sidebar.expander("Ajustes avançados (TVL)"):
    tvl_factor = st.slider("Peso do TVL no ajuste (1.00 = TVL integral)", 
                           min_value=0.0, max_value=5.0, value=1.0, step=0.05)

uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX (colunas: Date, Daily Revenue, MCAP, TVL opcional)", type=["csv","xlsx"])
protocol_name = st.sidebar.text_input("Nome do protocolo", value="protocolo1")
save_btn = st.sidebar.button("💾 Salvar protocolo no histórico")

# ---------------- MAIN ----------------
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        # tenta ; primeiro (padrão do usuário); se falhar, cai para vírgula
        try:
            df = pd.read_csv(uploaded_file, sep=";")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = compute_swpe(df, mode=mode, calc_mode=calc_mode, tvl_factor=tvl_factor)

    if not df.empty:
        # KPIs topo
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("SWPE Atual", f"{df['SWPE'].iloc[-1]:.3f}")
        col2.metric("Média SWPE", f"{df['SWPE'].mean():.3f}")
        col3.metric("Mediana SWPE", f"{df['SWPE'].median():.3f}")
        col4.metric("Receita EMA-7",  f"{df['EMA_7'].iloc[-1]:,.0f} USD")
        col5.metric("Receita EMA-14", f"{df['EMA_14'].iloc[-1]:,.0f} USD")
        col6.metric("Receita EMA-30", f"{df['EMA_30'].iloc[-1]:,.0f} USD")

        # Plot
        subtitle = f"{mode.upper()} • {calc_mode}"
        if calc_mode == "MCAP - TVL":
            subtitle += f" • TVL×{tvl_factor:.2f}"
        st.plotly_chart(plot_swpe(df, f"SWPE (DillaStyle) — {subtitle}"), use_container_width=True)

        # Tabela
        show_cols = ["Date","rev_daily","rev_annual","mcap","num_used","SWPE","EMA_7","EMA_14","EMA_30"]
        if "tvl" in df.columns:
            show_cols.insert(4, "tvl")
        st.dataframe(
            df[show_cols].rename(columns={
                "rev_daily":"Revenue (daily)",
                "rev_annual":"Revenue (annualizada)",
                "mcap":"MCAP",
                "tvl":"TVL",
                "num_used":"Numerador (usado)"
            })
        )

        # Salvar histórico
        if save_btn:
            file_path = os.path.join(SAVE_FOLDER, f"{protocol_name}_{mode}_{calc_mode.replace(' ','_')}_tvl{int(tvl_factor*100)}.csv")
            df.to_csv(file_path, index=False)
            st.success(f"Protocolo salvo em {file_path}")

# ---------------- LOAD & DELETE ----------------
st.sidebar.subheader("📚 Protocolos Salvos")
files = os.listdir(SAVE_FOLDER)
if files:
    choice = st.sidebar.selectbox("Selecione um protocolo", files)
    if choice:
        df_loaded = pd.read_csv(os.path.join(SAVE_FOLDER, choice))
        # Recalcula com os parâmetros atuais (modo/TVL) para consistência visual
        df_loaded = compute_swpe(df_loaded, mode=mode, calc_mode=calc_mode, tvl_factor=tvl_factor)
        if not df_loaded.empty:
            subtitle = f"{mode.upper()} • {calc_mode}"
            if calc_mode == "MCAP - TVL":
                subtitle += f" • TVL×{tvl_factor:.2f}"
            st.plotly_chart(plot_swpe(df_loaded, f"Reaberto {choice} — {subtitle}"), use_container_width=True)

        delete_btn = st.sidebar.button("🗑️ Excluir protocolo selecionado")
        if delete_btn:
            os.remove(os.path.join(SAVE_FOLDER, choice))
            st.sidebar.success(f"{choice} excluído com sucesso. Recarregue a página.")

