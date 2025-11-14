# app_funda_only_yahoo.py
import os, io, time, math, requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Yahoo fundamentals via yahooquery (plus riche que yfinance)
try:
    from yahooquery import Ticker
except ImportError:
    Ticker = None

# ------------------------
# SETUP
# ------------------------
st.set_page_config(page_title="S&P 500 — Fundamentals Only (Yahoo)", layout="wide")
st.title("📘 S&P 500 — Fundamentals Only (Yahoo Finance)")

load_dotenv()  # pas de clé requise pour Yahoo, mais on garde la compatibilité

if Ticker is None:
    st.error("Le paquet 'yahooquery' n'est pas installé.\n\n"
             "Installe :  \n"
             "`pip install yahooquery streamlit pandas numpy python-dotenv requests lxml`")
    st.stop()

# ------------------------
# HELPERS
# ------------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK_CSVS = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

@st.cache_data(ttl=3600)
def fetch_sp500_from_wikipedia():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        if "Symbol" in t.columns:
            t["Symbol"] = t["Symbol"].astype(str).str.strip()
            t["Symbol"] = t["Symbol"].str.replace(r"[^\w\.\-]", "", regex=True)
            return t[["Symbol","Security","GICS Sector","GICS Sub-Industry"]].rename(columns={
                "Symbol":"ticker","Security":"name","GICS Sector":"sector","GICS Sub-Industry":"industry"
            })
    raise RuntimeError("Table S&P 500 non trouvée sur Wikipédia.")

@st.cache_data(ttl=3600)
def fetch_sp500_fallback():
    headers = {"User-Agent":"Mozilla/5.0"}
    last_err = None
    for url in FALLBACK_CSVS:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            colmap = {}
            for c in df.columns:
                lc = c.lower()
                if lc.startswith("symbol"):
                    colmap[c] = "ticker"
                elif lc.startswith("security") or lc.startswith("name"):
                    colmap[c] = "name"
                elif "sector" in lc:
                    colmap[c] = "sector"
                elif "sub" in lc and "industry" in lc:
                    colmap[c] = "industry"
            df = df.rename(columns=colmap)
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.strip()
                keep = [c for c in ["ticker","name","sector","industry"] if c in df.columns]
                return df[keep]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Fallback S&P 500 échoué. Dernière erreur: {last_err}")

def fetch_sp500_universe():
    try:
        return fetch_sp500_from_wikipedia()
    except Exception as e:
        st.warning(f"Wikipédia indisponible/bloqué ({e}). Utilisation d’une source de secours.")
        return fetch_sp500_fallback()

def pctile_or_neutral(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s2.dropna().empty:
        return pd.Series(50.0, index=s.index)  # neutre si colonne vide
    return 100 * s2.fillna(s2.median()).rank(pct=True)

# ------------------------
# RÉCUP FUNDAMENTAUX YAHOO (yahooquery)
# ------------------------
YQ_CHUNK = 40  # batch pour éviter rate-limit

# Map Yahoo -> nos colonnes
YQ_FIELDS = {
    # Qualité
    "grossMargins": ("quality", "gross_margin"),            # marge brute %
    "profitMargins": ("quality", "net_margin"),             # marge nette %
    "returnOnEquity": ("quality", "roe"),                   # ROE %
    # Croissance
    "revenueGrowth": ("growth", "revenue_growth"),          # YoY
    "earningsQuarterlyGrowth": ("growth", "eps_growth"),    # proxy EPS growth
    # Solidité
    "debtToEquity": ("safety", "debt_to_equity"),           # %
    "currentRatio": ("safety", "current_ratio"),            # ratio
    "quickRatio": ("safety", "quick_ratio"),
    # Bonus info
    "trailingPE": ("info", "pe"),
    "priceToBook": ("info", "pb"),
    "beta": ("info", "beta"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Récupère par batch via yahooquery:
      - financial_data
      - summary_detail
      - key_stats  (en renfort si dispo)
    Renvoie 1 ligne par ticker avec nos colonnes cibles.
    """
    all_rows = []
    for i in range(0, len(tickers), YQ_CHUNK):
        batch = tickers[i:i+YQ_CHUNK]
        t = Ticker(batch, asynchronous=True)

        fd = t.financial_data or {}
        sd = t.summary_detail or {}
        ks = t.key_stats or {}

        for sym in batch:
            row = {"ticker": sym}
            def g(key):
                v = None
                if isinstance(fd.get(sym), dict):
                    v = fd[sym].get(key, None)
                if v is None and isinstance(sd.get(sym), dict):
                    v = sd[sym].get(key, None)
                if v is None and isinstance(ks.get(sym), dict):
                    v = ks[sym].get(key, None)
                return v

            for yq_key, (_, our_name) in YQ_FIELDS.items():
                row[our_name] = g(yq_key)

            all_rows.append(row)

        time.sleep(0.4)  # pause douce

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df

# ------------------------
# SCORING (score en 1ère colonne)
# ------------------------
def compute_score_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Sous-scores : listes de colonnes
    q_cols = ["gross_margin", "net_margin", "roe"]
    g_cols = ["revenue_growth", "eps_growth"]

    # D/E inversé (plus bas = mieux)
    inv_de = -pd.to_numeric(d["debt_to_equity"], errors="coerce")
    d["_inv_de"] = inv_de

    # Percentiles avec neutralisation si colonne vide
    q = [pctile_or_neutral(pd.to_numeric(d[c], errors="coerce")) for c in q_cols]
    quality = sum(q) / len(q)

    g = [pctile_or_neutral(pd.to_numeric(d[c], errors="coerce")) for c in g_cols]
    growth = sum(g) / len(g)

    s = [pctile_or_neutral(d["_inv_de"]),
         pctile_or_neutral(pd.to_numeric(d["current_ratio"], errors="coerce"))]
    safety = sum(s) / len(s)

    score = 0.40 * quality + 0.35 * growth + 0.25 * safety

    out = pd.DataFrame({
        "score": score.round(2),  # ✅ score en premier
        "ticker": d["ticker"],
        "quality": quality.round(2),
        "growth": growth.round(2),
        "safety": safety.round(2),
        # Champs bruts utiles
        "gross_margin": d["gross_margin"],
        "net_margin": d["net_margin"],
        "roe": d["roe"],
        "revenue_growth": d["revenue_growth"],
        "eps_growth": d["eps_growth"],
        "debt_to_equity": d["debt_to_equity"],
        "current_ratio": d["current_ratio"],
        "quick_ratio": d.get("quick_ratio", np.nan),
        "pe": d.get("pe", np.nan),
        "pb": d.get("pb", np.nan),
        "beta": d.get("beta", np.nan),
    })
    return out

# ------------------------
# UI
# ------------------------
col1, col2, col3 = st.columns(3)
with col1:
    min_score = st.slider("Score minimal", 0, 100, 50, 1)
with col2:
    limit_names = st.number_input("Limiter l'univers (0 = tous)", min_value=0, value=100, step=50)
with col3:
    show_raw = st.checkbox("Afficher un échantillon brut", value=False)

run_btn = st.button("🔎 Lancer l’analyse fondamentale (Yahoo)")

# ------------------------
# MAIN
# ------------------------
if run_btn:
    with st.spinner("Chargement de l'univers S&P 500..."):
        universe = fetch_sp500_universe()

    if limit_names and limit_names > 0:
        universe = universe.head(int(limit_names))
    st.write(f"Univers chargé : **{len(universe)}** tickers")

    tickers = universe["ticker"].tolist()

    with st.spinner("Récupération des fondamentaux Yahoo (batchs)..."):
        funda = fetch_yahoo_fundamentals(tickers)

    st.write(f"📦 Fondamentaux récupérés via Yahoo : **{len(funda)}** / {len(universe)}")

    if len(funda) == 0:
        st.warning("Aucune donnée récupérée via Yahoo. Réduis le nombre de tickers ou réessaie plus tard.")
        st.stop()

    if show_raw:
        st.subheader("Échantillon brut (Yahoo)")
        st.dataframe(funda.head(10))

    scored = compute_score_yahoo(funda)

    # Merge pour récupérer secteur/industrie et calculer les moyennes par secteur
    merged = universe.merge(scored, on="ticker", how="inner")
    out = merged[merged["score"] >= min_score].sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------
    # 🧮 Moyennes par secteur
    # ------------------------
    # On agrège sur quelques colonnes clés
    sector_cols = [
        "score", "quality", "growth", "safety",
        "gross_margin", "net_margin", "roe",
        "revenue_growth", "eps_growth",
        "debt_to_equity", "current_ratio"
    ]
    # Moyennes pondérées simples (arithmétiques)
    sector_summary = (
        merged[["sector"] + sector_cols]
        .groupby("sector", dropna=False)
        .mean(numeric_only=True)
        .round(2)
        .sort_values("score", ascending=False)
        .reset_index()
    )

    st.subheader("📋 Résultats (Yahoo fondamentaux)")
    st.dataframe(out)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Exporter CSV — Résultats",
            data=out.to_csv(index=False),
            file_name="sp500_fundamentals_yahoo_results.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "📥 Exporter CSV — Moyennes par secteur",
            data=sector_summary.to_csv(index=False),
            file_name="sp500_fundamentals_yahoo_sector_summary.csv",
            mime="text/csv"
        )

    st.subheader("🏷️ Moyennes par secteur (score, sous-scores, ratios)")
    st.dataframe(sector_summary)
