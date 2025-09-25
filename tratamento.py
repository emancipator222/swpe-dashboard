import streamlit as st
import pandas as pd
import numpy as np
import io, re

st.set_page_config(page_title="CSV → XLSX (Formato SWPE)", layout="wide")
st.title("Transformador: 3 CSVs → XLSX (Formato SWPE)")

# ----------------- Helpers -----------------
def pick_col(df, patterns):
    for p in patterns:
        for c in df.columns:
            if re.search(p, str(c), flags=re.I):
                return c
    return None

def parse_date_series(s: pd.Series) -> pd.Series:
    # Inteiro => timestamp (auto-detect ms vs s). String => to_datetime
    if np.issubdtype(s.dtype, np.number):
        med = pd.Series(s.dropna()).median()
        unit = "ms" if med and med > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce").dt.tz_localize(None).dt.normalize()
    else:
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def normalize_numeric(col: pd.Series) -> pd.Series:
    if np.issubdtype(col.dtype, np.number):
        return pd.to_numeric(col, errors="coerce")
    s = col.astype(str).str.strip()
    # Heurística: muitos ',' e poucos '.' => formato europeu ("1.234.567,89")
    comma = s.str.count(",").mean()
    dot = s.str.count(r"\.").mean()
    if comma > dot:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def dedupe_daily(df, date_col, val_col):
    # 1 linha por dia: manter o último valor do dia
    df = df[[date_col, val_col]].copy()
    df[date_col] = parse_date_series(df[date_col])
    df[val_col] = normalize_numeric(df[val_col])
    df = df.sort_values(date_col)
    return df.groupby(date_col, as_index=False)[val_col].last()

def transform(rev_df, tvl_df, mcap_df, join_mode="tvl_left"):
    # detectar colunas
    r_date = pick_col(rev_df, [r"^date$", "time", "timestamp"]) or "Date"
    r_val  = pick_col(rev_df, ["rev", "fee"]) or rev_df.columns[-1]

    t_date = pick_col(tvl_df, [r"^date$", "time", "timestamp"]) or "Date"
    t_val  = pick_col(tvl_df, ["tvl"]) or tvl_df.columns[-1]

    m_date = pick_col(mcap_df, [r"^date$", "time", "timestamp"]) or "Date"
    m_val  = pick_col(mcap_df, ["mcap", "market.?cap", "token.?mcap", "float.?cap"]) or mcap_df.columns[-1]

    # dedupe 1/dia
    R = dedupe_daily(rev_df, r_date, r_val).rename(columns={r_date: "Date", r_val: "Daily Revenue"})
    T = dedupe_daily(tvl_df, t_date, t_val).rename(columns={t_date: "Date", t_val: "TVL"})
    M = dedupe_daily(mcap_df, m_date, m_val).rename(columns={m_date: "Date", m_val: "MCAP"})

    # merge
    if join_mode == "inner":
        base = T.merge(M, on="Date", how="inner").merge(R, on="Date", how="inner")
    elif join_mode == "revenue_left":
        base = R.merge(T, on="Date", how="left").merge(M, on="Date", how="left")
    elif join_mode == "mcap_left":
        base = M.merge(T, on="Date", how="left").merge(R, on="Date", how="left")
    else:  # "tvl_left" (padrão do seu .xlsx de ref)
        base = T.merge(M, on="Date", how="left").merge(R, on="Date", how="left")

    # annualized e ordenação
    base["Annualized Revenue"] = (base["Daily Revenue"] * 365).round()
    base = base[["Date", "Daily Revenue", "Annualized Revenue", "MCAP", "TVL"]]
    # remover linhas totalmente vazias de MCAP+Revenue (ex.: TVL tem 1 dia a mais)
    drop_mask = base[["Daily Revenue", "MCAP"]].isna().all(axis=1)
    base = base[~drop_mask].copy()

    base = base.sort_values("Date", ascending=False).reset_index(drop=True)
    return base, R, T, M

def describe(name, df):
    if df.empty:
        return f"**{name}**: vazio"
    return (
        f"**{name}** — linhas: {len(df)} | "
        f"min: {pd.to_datetime(df['Date']).min().date()} | "
        f"max: {pd.to_datetime(df['Date']).max().date()}"
    )


# ----------------- UI -----------------
left, right = st.columns([1,1])

with left:
    rev_file = st.file_uploader("Revenue CSV", type=["csv"], key="rev")
    tvl_file = st.file_uploader("TVL CSV", type=["csv"], key="tvl")
    mcap_file = st.file_uploader("MCAP CSV", type=["csv"], key="mcap")

with right:
    join_mode = st.selectbox(
        "Base do merge",
        options=["tvl_left", "revenue_left", "mcap_left", "inner"],
        index=0,
        help="Para espelhar o .xlsx de referência, use 'tvl_left'."
    )

st.markdown("---")

if rev_file and tvl_file and mcap_file:
    # leitura crua
    rev_df  = pd.read_csv(rev_file)
    tvl_df  = pd.read_csv(tvl_file)
    mcap_df = pd.read_csv(mcap_file)

    final_df, R, T, M = transform(rev_df, tvl_df, mcap_df, join_mode=join_mode)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas (final)", len(final_df))
    c2.metric("Começa em", str(final_df["Date"].max().date()))
    c3.metric("Termina em", str(final_df["Date"].min().date()))
    c4.metric("Base do merge", join_mode)

    st.caption(describe("Revenue (1/dia)", R))
    st.caption(describe("TVL (1/dia)", T))
    st.caption(describe("MCAP (1/dia)", M))

    st.subheader("Prévia do arquivo final")
    st.dataframe(final_df.head(25), use_container_width=True)

    # Download XLSX em memória
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="SWPE")
    except Exception:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="SWPE")

    st.download_button(
        "📥 Baixar XLSX tratado",
        data=buf.getvalue(),
        file_name="SWPE_tratado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
else:
    st.info("Anexe os **3 CSVs** (Revenue, TVL, MCAP) para processar.")
