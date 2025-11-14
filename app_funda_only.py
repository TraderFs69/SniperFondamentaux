import os
import time
import io
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

load_dotenv()
POLY = os.getenv("POLYGON_API_KEY")

if not POLY:
    st.error("⚠️ POLYGON_API_KEY manquant. Crée un fichier .env (voir .env.example) puis relance l'app.")
    st.stop()

# ------------------------
# HELPERS
# ------------------------
def api_get(url: str, params: dict = None, tries: int = 3, sleep: float = 0.8):
    """Requête GET avec reprise automatique."""
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
    """Accès sécurisé dans un dict imbriqué."""
    if d is None:
        return default
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def normalize_percentile(s: pd.Series):
    return 100 * s.rank(pct=True)

# ------------------------
# S&P 500 UNIVERSE (robuste)
# ------------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK_CSVS = [
    # Fallbacks publics maintenus (si Wikipédia bloque)
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

@st.cache_data(ttl=3600)
def fetch_sp500_from_wikipedia():
    """Tente Wikipédia avec User-Agent; si échec, déclenche une exception capturée en amont."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()  # Provoque HTTPError si bloqué
    # read_html depuis le contenu récupéré (évite lecture directe par URL)
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        if "Symbol" in t.columns:
            t["Symbol"] = t["Symbol"].astype(str).str.strip()
            t["Symbol"] = t["Symbol"].str.replace(r"[^\w\.\-]", "", regex=True)
            return t[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
                columns={
                    "Symbol": "ticker",
                    "Security": "name",
                    "GICS Sector": "sector",
                    "GICS Sub-Industry": "industry",
                }
            )
    raise RuntimeError("Table du S&P 500 introuvable dans la page Wikipédia.")

@st.cache_data(ttl=3600)
def fetch_sp500_fallback():
    """Essaie des CSV publics si Wikipédia est bloqué."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_err = None
    for url in FALLBACK_CSVS:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Harmonisation des noms de colonnes potentiels
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
            keep = [c for c in ["ticker", "name", "sector", "industry"] if c in df.columns]
            if "ticker" in keep:
                df["ticker"] = df["ticker"].astype(str).str.strip()
                return df[keep]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Impossible de charger l'univers S&P 500 via les fallbacks. Dernière erreur: {last_err}")

def fetch_sp500_universe():
    """Stratégie: Wikipédia -> fallback(s)."""
    try:
        return fetch_sp500_from_wikipedia()
    except Exception as e:
        st.warning(f"Wikipédia indisponible/bloqué ({e}). Bascule vers une source de secours.")
        return fetch_sp500_fallback()

# ------------------------
# POLYGON FUNDAMENTALS
# ------------------------
def polygon_financials_ttm(ticker: str, limit=4):
    """Récupère les derniers fondamentaux (vX/reference/financials)."""
    url = "https://api.polygon.io/vX/reference/financials"
    params = {"ticker": ticker, "limit": limit, "apiKey": POLY}
    js = api_get(url, params=params)
    if not js or "results" not in js:
        return None
    results = js.get("results", [])
    if not results:
        return None

    # Tri par date de fin de période (desc)
    rows = []
    for r in results:
        period_end = get_in(r, "fiscal_period_end", None) or get_in(r, "end_date", None)
        if period_end:
            rows.append((period_end, r))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    ordered = [r for _, r in rows]

    def block(rec):
        return dict(
            gross_margin=get_in(rec, "ratios.gross_margin"),
            net_margin=get_in(rec, "ratios.net_margin"),
            roic=get_in(rec, "ratios.roic"),
            revenue_ttm=get_in(rec, "ttm.revenue"),
            eps_ttm=get_in(rec, "ttm.eps"),
            debt_to_equity=get_in(rec, "ratios.debt_to_equity"),
            interest_coverage=get_in(rec, "ratios.interest_coverage"),
        )

    latest = block(ordered[0])
    prev = block(ordered[1]) if len(ordered) > 1 else {k: np.nan for k in latest.keys()}

    return {
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

# ------------------------
# SCORING
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
    safety = (normalize_percentile((-d["debt_to_equity"]).replace([np.inf, -np.inf], np.nan).fillna(0))
              + pctile("interest_coverage")) / 2.0

    score = 0.40 * quality + 0.35 * growth + 0.25 * safety
    out = d[["ticker"]].copy()
    out["quality"] = quality
    out["growth"] = growth
    out["safety"] = safety
    out["score"] = score
    return out

# ------------------------
# UI
# ------------------------
col1, col2 = st.columns(2)
with col1:
    min_score = st.slider("Score minimal", 0, 100, 60, 1)
with col2:
    limit_names = st.number_input("Limiter l'univers (0 = tous)", min_value=0, value=50, step=50)

run_btn = st.button("🔎 Lancer l’analyse fondamentale")

# ------------------------
# MAIN
# ------------------------
if run_btn:
    with st.spinner("Chargement de l'univers S&P 500..."):
        universe_df = fetch_sp500_universe()

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

    # Filtrage + affichage
    out = merged[merged["score"] >= min_score].sort_values("score", ascending=False).reset_index(drop=True)

    st.subheader("📋 Résultats (fondamentaux uniquement)")
    st.dataframe(out)

    st.download_button(
        "📥 Exporter CSV",
        data=out.to_csv(index=False),
        file_name="sp500_fundamentals_polygon.csv",
        mime="text/csv"
    )
