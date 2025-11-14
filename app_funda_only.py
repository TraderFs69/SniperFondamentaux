import os
import time
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import streamlit as st

# ------------------------
# SETUP
# ------------------------
st.set_page_config(page_title="S&P 500 — Fundamentals Only (Polygon)", layout="wide")
st.title("📘 S&P 500 — Fundamentals Only (Polygon)")

# Charge la clé API Polygon depuis le fichier .env
load_dotenv()
POLY = os.getenv("POLYGON_API_KEY")

if not POLY:
    st.error("⚠️ POLYGON_API_KEY manquant. Crée un fichier .env (voir .env.example) puis relance l'app.")
    st.stop()

# ------------------------
# FONCTIONS UTILES
# ------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sp500_from_wikipedia():
    """
    Va chercher la liste officielle des compagnies du S&P 500 sur Wikipédia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    for t in tables:
        if "Symbol" in t.columns:
            t["Symbol"] = t["Symbol"].astype(str).str.strip()
            t["Symbol"] = t["Symbol"].str.replace(r"[^\w\.\-]", "", regex=True)
            return t[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry"
            })
    raise RuntimeError("Impossible de trouver la table du S&P 500 sur Wikipédia.")

def api_get(url: str, params: dict = None, tries: int = 3, sleep: float = 0.8):
    """Requête GET avec reprise automatique si échec"""
    params = params or {}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503):
                time.sleep(sleep * (i + 1))
                continue
        except requests.RequestException:
            time.sleep(sleep * (i + 1))
    return None

def get_in(d: dict, path: str, default=np.nan):
    """Accès sécurisé à une clé imbriquée dans un dictionnaire JSON"""
    if d is None:
        return default
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def normalize_percentile(s: pd.Series):
    """Convertit une série en percentiles (0–100)"""
    return 100 * s.rank(pct=True)

# ------------------------
# RÉCUPÉRATION DES FONDAMENTAUX POLYGON
# ------------------------
def polygon_financials_ttm(ticker: str, limit=4):
    """
    Récupère les derniers états financiers d'une compagnie depuis Polygon.
    Utilise le endpoint vX/reference/financials.
    """
    url = "https://api.polygon.io/vX/reference/financials"
    params = {"ticker": ticker, "limit": limit, "apiKey": POLY}
    js = api_get(url, params=params)
    if not js or "results" not in js:
        return None
    results = js.get("results", [])
    if not results:
        return None

    # Trie les périodes par date
    rows = []
    for r in results:
        period_end = get_in(r, "fiscal_period_end", None) or get_in(r, "end_date", None)
        if period_end:
            rows.append((period_end, r))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    ordered = [r for _, r in rows]

    def extract_block(rec):
        gross_margin = get_in(rec, "ratios.gross_margin")
        net_margin = get_in(rec, "ratios.net_margin")
        roic = get_in(rec, "ratios.roic")
        revenue_ttm = get_in(rec, "ttm.revenue")
        eps_ttm = get_in(rec, "ttm.eps")
        debt_to_equity = get_in(rec, "ratios.debt_to_equity")
        interest_coverage = get_in(rec, "ratios.interest_coverage")
        return dict(
            gross_margin=gross_margin,
            net_margin=net_margin,
            roic=roic,
            revenue_ttm=revenue_ttm,
            eps_ttm=eps_ttm,
            debt_to_equity=debt_to_equity,
            interest_coverage=interest_coverage,
        )

    latest = extract_block(ordered[0])
    prev = extract_block(ordered[1]) if len(ordered) > 1 else {k: np.nan for k in latest.keys()}

    out = {
        "ticker": ticker,
        "roic": latest["roic"],
        "gross_margin": latest["gross_margin"],
        "net_margin": latest["net_margin"],
        "revenue_ttm": latest["revenue_ttm"],
        "revenue_ttm_prev": prev["revenue_ttm"],
        "eps_ttm": latest["eps_ttm"],
        "eps_ttm_prev": prev["eps_ttm"],
        "debt_to_equity": latest["debt_to_equity"],
        "interest_coverage": latest["interest_coverage"],
    }
    return out

# ------------------------
# CALCUL DU SCORE FONDAMENTAL
# ------------------------
def compute_fundamental_score(df: pd.DataFrame):
    d = df.copy()
    for col in [
        "roic", "gross_margin", "net_margin", "revenue_ttm",
        "revenue_ttm_prev", "eps_ttm", "eps_ttm_prev",
        "debt_to_equity", "interest_coverage"
    ]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Croissance TTM
    d["rev_growth"] = (d["revenue_ttm"] - d["revenue_ttm_prev"]) / d["revenue_ttm_prev"]
    d["eps_growth"] = (d["eps_ttm"] - d["eps_ttm_prev"]) / d["eps_ttm_prev"]

    def pctile(col):
        s = d[col].replace([np.inf, -np.inf], np.nan)
        return normalize_percentile(s.fillna(s.median()))

    quality = (pctile("roic") + pctile("gross_margin") + pctile("net_margin")) / 3.0
    growth = (pctile("rev_growth") + pctile("eps_growth")) / 2.0
    safety = (
        normalize_percentile((-d["debt_to_equity"]).replace([np.inf, -np.inf], np.nan).fillna(0))
        + pctile("interest_coverage")
    ) / 2.0

    score = 0.40 * quality + 0.35 * growth + 0.25 * safety
    out = d[["ticker"]].copy()
    out["quality"] = quality
    out["growth"] = growth
    out["safety"] = safety
    out["score"] = score
    return out

# ------------------------
# INTERFACE UTILISATEUR
# ------------------------
col1, col2 = st.columns(2)
with col1:
    min_score = st.slider("Score minimal", 0, 100, 60, 1)
with col2:
    limit_names = st.number_input("Limiter l'univers (0 = tous)", min_value=0, value=50, step=50)

run_btn = st.button("🔎 Lancer l’analyse fondamentale")

# ------------------------
# LOGIQUE PRINCIPALE
# ------------------------
if run_btn:
    with st.spinner("Récupération de la liste S&P 500 depuis Wikipédia..."):
        universe_df = fetch_sp500_from_wikipedia()

    if limit_names and limit_names > 0:
        universe_df = universe_df.head(int(limit_names))

    st.write(f"Univers chargé : **{len(universe_df)}** tickers")

    fin_rows = []
    progress = st.progress(0)
    tickers = universe_df["ticker"].tolist()

    for i, t in enumerate(tickers):
        progress.progress((i + 1) / len(tickers))
        try:
            row = polygon_financials_ttm(t)
            if row:
                fin_rows.append(row)
        except Exception:
            continue

    if not fin_rows:
        st.warning("Aucun fondamental récupéré depuis Polygon. Vérifie ton plan/endpoint.")
        st.stop()

    raw_df = pd.DataFrame(fin_rows)
    scored = compute_fundamental_score(raw_df)
    merged = universe_df.merge(scored, on="ticker", how="inner")

    # Filtrage
    out = merged[merged["score"] >= min_score].sort_values("score", ascending=False).reset_index(drop=True)

    st.subheader("📋 Résultats (fondamentaux uniquement)")
    st.dataframe(out)

    st.download_button(
        "📥 Exporter CSV",
        data=out.to_csv(index=False),
        file_name="sp500_fundamentals_polygon.csv",
        mime="text/csv"
    )
