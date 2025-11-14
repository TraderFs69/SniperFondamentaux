import os, io, time, requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ------------------------
# SETUP
# ------------------------
st.set_page_config(page_title="S&P 500 — Fundamentals Only (FMP)", layout="wide")
st.title("📘 S&P 500 — Fundamentals Only (Financial Modeling Prep)")

load_dotenv()
FMP = os.getenv("FMP_API_KEY", "")

if not FMP:
    st.warning("ℹ️ Aucun FMP_API_KEY détecté dans `.env`. Certaines requêtes FMP fonctionnent sans clé mais sont limitées. "
               "Idéalement, ajoute FMP_API_KEY=ta_clef_dans_.env (clé gratuite dispo sur financialmodelingprep.com).")

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

def api_json(url, params=None, tries=3, sleep=0.7):
    params = params or {}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503):
                time.sleep(sleep*(i+1))
                continue
        except requests.RequestException:
            time.sleep(sleep*(i+1))
    return None

# ------------------------
# FUNDAMENTALS VIA FMP
# ------------------------
def fmp_ratios_ttm(ticker: str):
    url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}"
    params = {"apikey": FMP} if FMP else {}
    js = api_json(url, params=params)
    if not js or not isinstance(js, list) or len(js)==0:
        return None
    rec = js[0]
    def g(k): return rec.get(k, np.nan)
    return dict(
        gross_margin=g("grossProfitMarginTTM"),
        net_margin=g("netProfitMarginTTM"),
        roic=g("returnOnInvestedCapitalTTM"),
        revenue_growth=g("revenueGrowthTTM"),
        eps_growth=g("epsGrowthTTM"),
        debt_to_equity=g("debtEquityTTM"),
        interest_coverage=g("interestCoverageTTM"),
    )

def fmp_key_metrics_ttm(ticker: str):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}"
    params = {"apikey": FMP} if FMP else {}
    js = api_json(url, params=params)
    if not js or not isinstance(js, list) or len(js)==0:
        return None
    rec = js[0]
    def g(k): return rec.get(k, np.nan)
    return dict(
        gross_margin=g("grossProfitMarginTTM"),
        net_margin=g("netProfitMarginTTM"),
        roic=g("roicTTM") or g("returnOnInvestedCapitalTTM"),
        revenue_growth=g("revenueGrowthTTM"),
        eps_growth=g("epsdilutedGrowthTTM") or g("epsGrowthTTM"),
        debt_to_equity=g("debtToEquityTTM") or g("debtEquityTTM"),
        interest_coverage=np.nan
    )

def get_funda_for(ticker: str):
    d = fmp_ratios_ttm(ticker)
    if d is None:
        d = {}
    km = fmp_key_metrics_ttm(ticker)
    if km:
        for k, v in km.items():
            if (k not in d) or (pd.isna(d[k])):
                d[k] = v
    if not d:
        return None
    d["ticker"] = ticker
    return d

# ------------------------
# SCORING
# ------------------------
def pctile_or_neutral(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s2.dropna().empty:
        return pd.Series(50.0, index=s.index)
    return 100 * s2.fillna(s2.median()).rank(pct=True)

def compute_score(df: pd.DataFrame):
    d = df.copy()
    q1 = pctile_or_neutral(d["roic"])
    q2 = pctile_or_neutral(d["gross_margin"])
    q3 = pctile_or_neutral(d["net_margin"])
    quality = (q1 + q2 + q3) / 3.0

    g1 = pctile_or_neutral(d["revenue_growth"])
    g2 = pctile_or_neutral(d["eps_growth"])
    growth = (g1 + g2) / 2.0

    s1 = pctile_or_neutral(-pd.to_numeric(d["debt_to_equity"], errors="coerce"))
    s2 = pctile_or_neutral(d["interest_coverage"])
    safety = (s1 + s2) / 2.0

    score = 0.40*quality + 0.35*growth + 0.25*safety

    out = pd.DataFrame({
        "ticker": d["ticker"],
        "quality": quality.round(2),
        "growth": growth.round(2),
        "safety": safety.round(2),
        "score": score.round(2),
        "roic": d["roic"],
        "gross_margin": d["gross_margin"],
        "net_margin": d["net_margin"],
        "revenue_growth": d["revenue_growth"],
        "eps_growth": d["eps_growth"],
        "debt_to_equity": d["debt_to_equity"],
        "interest_coverage": d["interest_coverage"],
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

run_btn = st.button("🔎 Lancer l’analyse fondamentale (FMP)")

# ------------------------
# MAIN
# ------------------------
if run_btn:
    with st.spinner("Chargement de l'univers S&P 500..."):
        universe = fetch_sp500_universe()

    if limit_names and limit_names > 0:
        universe = universe.head(int(limit_names))
    st.write(f"Univers chargé : **{len(universe)}** tickers")

    rows = []
    prog = st.progress(0)
    tickers = universe["ticker"].tolist()

    for i, t in enumerate(tickers):
        prog.progress((i+1)/len(tickers))
        try:
            d = get_funda_for(t)
            if d:
                rows.append(d)
        except Exception:
            continue

    st.write(f"📦 Fondamentaux récupérés via FMP : **{len(rows)}** / {len(universe)}")
    if len(rows) == 0:
        st.warning("Aucune donnée récupérée depuis FMP. Vérifie ta clé FMP_API_KEY (ou crée-en une gratuite), puis réessaie.")
        st.stop()

    raw = pd.DataFrame(rows)
    if show_raw:
        st.subheader("Échantillon brut (FMP)")
        st.dataframe(raw.head(10))

    scored = compute_score(raw)
    merged = universe.merge(scored, on="ticker", how="inner")
    out = merged[merged["score"] >= min_score].sort_values("score", ascending=False).reset_index(drop=True)

    st.subheader("📋 Résultats (FMP fondamentaux)")
    st.dataframe(out)

    st.download_button("📥 Exporter CSV",
                       data=out.to_csv(index=False),
                       file_name="sp500_fundamentals_fmp.csv",
                       mime="text/csv")
