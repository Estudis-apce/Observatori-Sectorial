# ---------------------------
# Standard library
# ---------------------------
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Union
import base64
import io
import json
import re

# ---------------------------
# Third-party libraries
# ---------------------------
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Importante: antes de pyplot en entornos sin display (ej. Streamlit)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.express as px
import plotly.graph_objects as go
import numpy_financial as npf

import geopandas as gpd

import yaml
from yaml.loader import SafeLoader

import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from streamlit_folium import folium_static


# ---------------------------
# ReportLab (PDF)
# ---------------------------
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from reportlab.platypus import (
    SimpleDocTemplate,
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,           # Nota: Image de platypus
    KeepTogether,
    PageBreak,
    NextPageTemplate,
)
# Alias útil para diferenciar imágenes si lo prefieres en tu código:
from reportlab.platypus import Image as RLImage

from reportlab.platypus.flowables import CondPageBreak

def auto_spinner(func):
    def wrapper(*args, **kwargs):
        with st.spinner("Carregant dades..."):
            return func(*args, **kwargs)
    return wrapper

# ========== COLORES / CONFIG ==========
# Paleta "Navy & Coral": proposta sense ancorar-se a cap direcció anterior. Primary
# (navy) i text (grafit) són colors DIFERENTS a propòsit (a "Ink & Amber" eren el
# mateix to i això trencava el contrast quan un pill seleccionat quedava amb fons i
# text idèntics sota una regla `!important` existent).
CSS_COLORS = {
    "bg": "#FAF9F6",
    "primary": "#2B3A67",
    "accent": "#D9773F",
    "text": "#262B36",
    "brand_dark": "#3F5C3F"
}

# ========== TEMA CLAR / FOSC ==========
# Font única de veritat de la paleta clar/fosc de l'app (web, en viu). Es
# reinjecta com a variables CSS a cada rerun (apply_theme_css) segons
# st.session_state["theme"]. Els valors "light" reprodueixen exactament
# CSS_COLORS perquè no hi hagi dues fonts de veritat pel mode clar.
# No afecta el PDF/Matplotlib (document estàtic descarregable amb la
# paleta de marca fixa, independent del mode clar/fosc de la sessió web).
LIGHT_THEME = {
    "bg": CSS_COLORS["bg"],
    "surface": "rgba(255, 255, 255, 0.55)",
    "surface-solid": "#ffffff",
    "table-alt": CSS_COLORS["accent"],
    "primary": CSS_COLORS["primary"],
    "primary-hover": CSS_COLORS["accent"],
    "accent": CSS_COLORS["accent"],
    "text": CSS_COLORS["text"],
    "text-inverse": "#ffffff",
    "border": "rgba(43, 58, 103, 0.16)",
    "border-strong": "rgba(43, 58, 103, 0.26)",
    "focus": "rgba(38, 43, 54, 0.35)",
    "shadow": "rgba(0, 0, 0, 0.12)",
}
DARK_THEME = {
    "bg": "#12151c",
    "surface": "rgba(255, 255, 255, 0.06)",
    "surface-solid": "#1b2030",
    "table-alt": "#3a2818",
    "primary": "#7c93c7",
    "primary-hover": "#28324a",
    "accent": "#28324a",
    "text": "#e7e9ee",
    "text-inverse": "#12151c",
    "border": "rgba(124, 147, 199, 0.28)",
    "border-strong": "rgba(124, 147, 199, 0.4)",
    "focus": "rgba(124, 147, 199, 0.4)",
    "shadow": "rgba(0, 0, 0, 0.5)",
}
THEMES = {"light": LIGHT_THEME, "dark": DARK_THEME}


def apply_theme_css(theme_name: str):
    """Injecta les variables CSS --app-* del tema seleccionat com a <style>
    addicional, sobreescrivint els valors per defecte de main.css. Tot
    main.css (fons, text, menús, botons, taules, selectors) ja consumeix
    aquestes variables, així que un sol punt d'injecció temeja tota l'app."""
    palette = THEMES.get(theme_name, LIGHT_THEME)
    vars_css = "; ".join(f"--app-{k}: {v}" for k, v in palette.items())
    st.markdown(
        f"<style>:root {{ color-scheme: {theme_name}; {vars_css}; }}</style>",
        unsafe_allow_html=True,
    )


def st_plotly_chart(fig, **kwargs):
    """Embolcall de st.plotly_chart que aplica el tema clar/fosc actual
    (colors de text, eixos i llegenda) al vol. Cal fer-ho aquí en comptes
    de dins de cada funció generadora de gràfic perquè moltes estan
    cachejades amb @st.cache_data sense el tema com a argument: mutar la
    figura ja retornada (còpia pròpia de cada crida, no l'objecte cachejat)
    just abans de pintar-la evita haver de tocar les ~130 crides existents."""
    palette = THEMES.get(st.session_state.get("theme", "light"), LIGHT_THEME)
    fig.update_layout(
        font=dict(color=palette["text"]),
        legend=dict(font=dict(color=palette["text"])),
        title=dict(font=dict(color=palette["text"])),
    )
    fig.update_xaxes(color=palette["text"], gridcolor=palette["border"], zerolinecolor=palette["border"])
    fig.update_yaxes(color=palette["text"], gridcolor=palette["border"], zerolinecolor=palette["border"])
    return st.plotly_chart(fig, **kwargs)


GLOBAL_PALETTE = {
    "total": "#2d538f",
    "segunda_ma": "#D9773F",
    "nou": "#1b7f3a",
    "unifamiliar": "#D9773F",
    "plurifamiliar": "#1b7f3a",
}

# Paletes específiques dels gràfics Plotly (colors propis, diferents dels de GLOBAL_PALETTE)
# 6 colors perquè els gràfics d'àrea amb 6 categories (superfície construïda:
# fins a 50m2 ... més de 150m2) no repeteixin color entre la 1a/5a i 2a/6a
# categoria (amb només 4 colors es confonien visualment).
PLOTLY_PALETTE = ["#2d538f", "#D9773F", "#3F5C3F", "#6B6B6B", "#7A5C8E", "#C9A227"]
PLOTLY_PALETTE_DEMOGRAFIA = ["#6495ED", "#7DF9FF", "#87CEEB", "#A7C7E7", "#FFA07A"]

# Noms llargs (catalá) de les variables d'idescat_muns / df_mun_idescat, usats a la
# pestanya "Altres indicadors" (Municipis) i a l'"Informe de mercat" del PDF.
# Únic punt de manteniment: abans hi havia aquest mateix diccionari duplicat als dos llocs.
NOMBRE_VARIABLES_IDESCAT = {
    "AfiliatSS_Agricultura": "Afiliats a la Seguretat Social – Agricultura",
    "AfiliatSS_Construcció": "Afiliats a la Seguretat Social – Construcció",
    "AfiliatSS_Indústria": "Afiliats a la Seguretat Social – Indústria",
    "AfiliatSS_Serveis": "Afiliats a la Seguretat Social – Serveis",
    "AfiliatSS_Total": "Afiliats a la Seguretat Social – Total",
    "Atur registrat_Agricultura": "Atur registrat – Agricultura",
    "Atur registrat_Construcció": "Atur registrat – Construcció",
    "Atur registrat_Indústria": "Atur registrat – Indústria",
    "Atur registrat_Serveis": "Atur registrat – Serveis",
    "Atur registrat_Total": "Atur registrat – Total",
    "IRPF_Base_imposable": "Base imposable mitjana de l’IRPF (€)",
    "Matrimonis_Total": "Nombre de matrimonis",
    "Naixements_Total": "Nombre de naixements",
    "Parc_vehicles_Total": "Parc total de vehicles",
    "Pensionistes_Total": "Nombre de pensionistes",
    "Residus_mun_per_capita": "Residus municipals per càpita (kg/hab/dia)",
    "poblacio_activa": "Població activa",
    "poblacio_ocupada": "Població ocupada",
    "poblacio_desocupada": "Població desocupada",
    "poblacio_inactiva": "Població inactiva",
    "Població total": "Població total",
    "Creixement població interanual": "Creixement interanual de la població",
    "Població 25–34 anys (% sobre total)": "Població de 25 a 34 anys (% sobre total)",
    "Població 35–44 anys (% sobre total)": "Població de 35 a 44 anys (% sobre total)",
    "Naixements sobre població": "Naixements sobre població total (%)",
    "Matrimonis sobre població": "Matrimonis sobre població total (%)",
}

TABLE_TRIM_START_YEAR = 2023
TABLE_ANNUAL_START_YEAR = 2014
SERIES_START_YEAR = 2014
TITLE_SPACING_CM = 0.6
BLOCK_SPACING_CM = 0.8
# Mida de pàgina del PDF de l'informe de mercat: panoràmica 16:9 (com una diapositiva
# de PowerPoint, 33,87 x 19,05 cm) en lloc de l'A4 apaïsat (29,7 x 21 cm, ràtio 1,41)
# que es feia servir abans i que quedava massa "quadrat".
PDF_PAGE_SIZE = (33.87 * cm, 19.05 * cm)
CURRENT_YEAR_LIMIT = 2026  # Límit superior (any) de les dades disponibles; únic punt a actualitzar cada any
# Distinció entre tres conceptes que sovint es confonen:
#  - datetime.now().year   -> l'any real d'avui (rellotge del sistema). Només s'ha
#    d'usar per a metadades (data de generació d'un informe, nom de fitxer), MAI
#    per decidir quines dades mostrar.
#  - CURRENT_YEAR_LIMIT     -> l'any més recent per al qual hi ha dades carregades
#    (es manté manualment, en aquesta constant, cada vegada que s'actualitzen les
#    dades — no depèn del rellotge).
#  - LAST_CLOSED_YEAR       -> l'últim any complet/tancat amb dades anuals fiables
#    (CURRENT_YEAR_LIMIT - 1). És el que s'ha d'usar per decidir quin és l'"any
#    anterior" a l'hora de triar entre dada anual tancada i estimació mensual/
#    trimestral parcial (indicator_year, gràfics de barres anuals, KPIs del PDF).
LAST_CLOSED_YEAR = CURRENT_YEAR_LIMIT - 1

# ========== VIABILITAT DE PROMOCIÓ ==========
# Hipòtesis fixes replicades tal com estan a Viabilidad_promocion/APP_Dades.py
# (mateixos percentatges i estructura de capital validats en aquella app).
# No es toca res de Viabilidad_promocion; és només codi de referència.
VIAB_MAX_TRIM = 10  # nombre fix de trimestres (T0..T9)
VIAB_RECURSOS_PROPIS_PCT = 0.40  # 40% recursos propis / 60% crèdit sobre ingressos per vendes
VIAB_CREDIT_PCT = 0.60
VIAB_OTROS_SOLAR_PCT = 0.03
VIAB_HONORARIS_PCT = 0.07
VIAB_LLICENCIES_PCT = 0.05
VIAB_GASTOS_LEGALS_PCT = 0.02
VIAB_ALTRES_EDIF_PCT = 0.03
VIAB_ADMIN_PROMOCIO_PCT = 0.05
VIAB_COMERCIALITZACIO_PCT = 0.05
VIAB_IVA_SOLAR_PCT = 0.16
VIAB_IVA_EDIFICACIO_PCT = 0.07
VIAB_GASTOS_CONSTITUCIO_PCT = 0.01
VIAB_MIN_UNITATS_OFERTA = 5  # mínim d'habitatges nous en oferta (Atlas) per considerar el preu/m² representatiu

# ========== RUTES / FITXERS EXTERNS ==========
CSS_FILE = "main.css"
LOGO_APCE = "APCE_mod.png"
LOGO_APCE_WEB = "APCE_mod_transparent.png"
LOGO_APCE_WEB_DARK = "APCE_mod_dark.png"
LOGO_CLOSING = "APCE_serveis1.png"
SHAPEFILE_MUN = "shapefile_mun.geojson"
DATA_FILE_IDESCAT = "Idescat.json"
DATA_FILE_CENSO = "Censo2021.json"
DATA_FILE_INDICADORS_MUN = "Indicadors_mun.json"
DATA_FILE_SIMPLE = "DT_simple.json"
# Estudi d'Oferta de nova construcció: única font, l'Excel de l'Atlas (substitueix
# l'antic proveïdor DT_oferta_conjuntura.json + fulls històrics 2019-2025).
DATA_FILE_ATLAS_OFERTA = "BBDD_Atlas_trimmed.json"

# Informes sectorials APCE (PDF complet allotjat a apcebcn.cat): la imatge de portada
# (ja present a la carpeta del projecte) enllaça amb el PDF corresponent.
INFORMES_SECTORIALS = [
    {"any": 2024, "img": "Informe-sectorial-CATALUNYA_2024_FINAL.jpg", "url": "https://apcebcn.cat/wp-content/uploads/2025/07/Informe-Sectorial-2024.pdf"},
    {"any": 2023, "img": "Informe-sectorial-CATALUNYA_2023_FINAL.jpg", "url": "https://apcebcn.cat/wp-content/uploads/2024/08/Informe-Sectorial-2023.pdf"},
    {"any": 2022, "img": "Informe-sectorial-CATALUNYA_2022_FINAL.jpg", "url": "https://apcebcn.cat/wp-content/uploads/2023/07/informe-sectorial-2022.pdf"},
]
ATLAS_PERIODES = ["2025_H1", "2026_H1"]  # format correcte: "<any>_H1" (no "H1_<any>")

# ========== FORMATEO ==========
def _mpl_finish(fig) -> bytes:
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight", facecolor=CSS_COLORS["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _try_num_col(col):
    """Converteix una columna a numèric si es pot; si no, la deixa igual (substitueix
    pd.to_numeric(..., errors='ignore'), obsolet des de pandas 2.2)."""
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col

def _elementwise(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Aplica fn a cada element del DataFrame (substitueix DataFrame.applymap,
    obsolet des de pandas 2.1, mantenint compatibilitat amb pandas < 2.1)."""
    return df.apply(lambda col: col.map(fn))

def _format_thousands(x, pos=None):
    try:
        s = f"{x:,.0f}"
        return s.replace(",", ".")
    except Exception:
        return str(x)

def _format_df_thousands(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: f"{v:,.0f}".replace(",", ".") if pd.notnull(v) else "")
    return df2

def _delta_fmt(delta_str: Optional[str]) -> str:
    if not delta_str:
        return ""
    try:
        val = float(str(delta_str).replace("%", "").replace(",", "."))
    except Exception:
        return f"{delta_str}"
    arrow = "▲" if val >= 0 else "▼"
    color = "#1b7f3a" if val >= 0 else "#b00020"
    val_str = f"{abs(val):.1f}".replace(".", ",")
    return f"<font color='{color}'>{arrow} {val_str}%</font>"

# ========== FORMATO NUMÉRICO ESPAÑOL (miles con punto, decimales con coma) ==========
# Únic punt de control del format numèric de tota l'app (mètriques, taules, PDF).
_ES_NUM_RE = re.compile(r"-?\d[\d.,]*")

def _es_num_str(s: str) -> str:
    """Passa els números d'una cadena de format anglosaxó (1,234.5) a espanyol (1.234,5).
    Només toca els números (regex), de manera que sufixos com '%' o 'p.b.' queden intactes."""
    def _swap(m):
        return m.group(0).replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return _ES_NUM_RE.sub(_swap, s)

def st_metric(label=None, value=None, delta=None, **kwargs):
    """Embolcall de st.metric que mostra els números en format espanyol.
    - cadenes: es converteixen amb _es_num_str (1,234.5 -> 1.234,5)
    - enters/decimals crus: s'afegeix el separador de milers amb punt."""
    if isinstance(value, str):
        value = _es_num_str(value)
    elif isinstance(value, (int, np.integer)):
        value = f"{int(value):,}".replace(",", ".")
    elif isinstance(value, (float, np.floating)):
        value = _es_num_str(f"{value}")
    if isinstance(delta, str):
        delta = _es_num_str(delta)
    return st.metric(label, value, delta, **kwargs)

def taula_html_es(df, precision=1) -> str:
    """HTML d'una taula (DataFrame) amb números en format espanyol (per a taules que
    no passen per format_dataframes, com les mensuals)."""
    return df.style.format(thousands=".", decimal=",", precision=precision).to_html()

# ========== ÍNDICES / PERIODOS / FILTROS ==========
def _flatten_period_token(token) -> str:
    if isinstance(token, str):
        s = token.upper().replace("Q", "T")
        if "T" in s:
            return s if s.startswith("T") else ("T" + s.split("T")[-1])
        if s.isdigit():
            return f"T{int(s)}"
        return s
    if isinstance(token, (int, np.integer)):
        return f"T{int(token)}"
    return str(token)

def _flatten_period_index(idx: Iterable) -> list:
    if isinstance(idx, pd.MultiIndex) and len(idx.levels) == 2:
        out = []
        for (y, t) in idx:
            try:
                y_str = str(int(y))
            except Exception:
                y_str = str(y)
            t_str = _flatten_period_token(t)
            out.append(f"{y_str}{t_str if t_str else ''}")
        return out

    if isinstance(idx, pd.DatetimeIndex):
        try:
            periods = idx.to_period("Q")
            return [f"{p.year}T{p.quarter}" for p in periods]
        except Exception:
            return [str(d.year) for d in idx]

    return [str(x) for x in idx]

def _infer_year_from_label(label: str) -> Optional[int]:
    try:
        return int(str(label)[:4])
    except Exception:
        return None

def _filter_df_by_year(df: pd.DataFrame, start_year: int = SERIES_START_YEAR) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        return data[data.index.year >= start_year]
    if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 1:
        try:
            years = data.index.get_level_values(0).astype(int)
            return data[years >= start_year]
        except Exception:
            pass
    try:
        idx_str = [str(i) for i in data.index]
        mask = []
        for lab in idx_str:
            y = _infer_year_from_label(lab)
            mask.append((y is None) or (y >= start_year))
        return data[np.array(mask)]
    except Exception:
        return data

# ========== MATPLOTLIB HELPERS ==========
def _mpl_base(figsize=(13, 6), dpi=190):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(CSS_COLORS["bg"])
    ax.set_facecolor(CSS_COLORS["bg"])
    ax.tick_params(colors=CSS_COLORS["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    return fig, ax

def _tune_axes(ax, max_xticks=6, force_all_xticks=False):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.grid(False)
    if not force_all_xticks:
        xs = ax.get_xticks()
        if len(xs) > max_xticks and max_xticks > 0:
            step = max(1, len(xs) // max_xticks)
            try:
                ax.set_xticks(xs[::step])
            except Exception:
                pass
    for label in ax.get_xticklabels():
        label.set_color(CSS_COLORS["text"])
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_color(CSS_COLORS["text"])
        label.set_fontsize(9)

def _annotate_last(ax, x_labels: list, y: np.ndarray):
    try:
        if len(y) == 0 or np.all(np.isnan(y)):
            return
        ax.annotate(
            f"{y[-1]:,.0f}".replace(",", "."),
            xy=(len(x_labels) - 1, y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=CSS_COLORS["text"]
        )
    except Exception:
        pass

def mpl_line(df: pd.DataFrame, cols: list, title: str, ylab: str,
             xlab: str = "Període", start_year: int = SERIES_START_YEAR,
             palette: Optional[List[str]] = None,
             force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"], "#727375"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)

    plotted = False
    for i, c in enumerate(sel):
        y = pd.to_numeric(data[c], errors="coerce").values
        ax.plot(x_labels, y, label=c, linewidth=1.8, color=palette[i % len(palette)])
        _annotate_last(ax, x_labels, y)
        plotted = True

    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.set_xlabel(xlab, color=CSS_COLORS["text"])
    ax.grid(False)
    if plotted:
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_bar(df: pd.DataFrame, cols: list, title: str, ylab: str,
            start_year: int = SERIES_START_YEAR,
            palette: Optional[List[str]] = None,
            force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df.copy(), start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"], "#727375"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)
    n = len(sel)
    if n == 0 or len(x_labels) == 0:
        return _mpl_finish(fig)

    width = 0.8 / n
    x_idx = np.arange(len(x_labels))
    cur = 0
    for i, c in enumerate(sel):
        y = pd.to_numeric(data[c], errors="coerce").values
        offs = x_idx + cur * width
        bars = ax.bar(
            offs, y, width=width, label=c,
            color=palette[i % len(palette)],
            edgecolor="white", linewidth=0.3
        )
        # Etiqueta de valor en cada barra
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:,.0f}".replace(",", "."),
                    ha="center", va="bottom", fontsize=8, color=CSS_COLORS["text"]
                )
        cur += 1

    ax.set_xticks(x_idx + (n - 1) * width / 2)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.grid(False)
    if n > 1:  # leyenda solo si hay >1 serie
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_area(df: pd.DataFrame, cols: List[str], title: str, ylab: str,
             xlab: str = "Període", start_year: int = SERIES_START_YEAR,
             palette: Optional[List[str]] = None,
             force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = ["#2d538f", "#1b7f3a", "#D9773F", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)
    if sel:
        ys = [pd.to_numeric(data[c], errors="coerce").values for c in sel]
        ax.stackplot(x_labels, ys, labels=sel, colors=[palette[i % len(palette)] for i in range(len(sel))], alpha=0.95)

    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.set_xlabel(xlab, color=CSS_COLORS["text"])
    ax.grid(False)
    if sel:
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_dual_line(df: pd.DataFrame, left_col: str, right_col: str,
                  left_label: str, right_label: str,
                  left_ylab: str, right_ylab: str,
                  start_year: int = SERIES_START_YEAR,
                  left_color: str = GLOBAL_PALETTE["total"],
                  right_color: str = GLOBAL_PALETTE["segunda_ma"],
                  force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax1 = _mpl_base()
    x_labels = _flatten_period_index(data.index)

    y1 = pd.to_numeric(data.get(left_col, pd.Series(index=data.index)), errors="coerce").values
    y2 = pd.to_numeric(data.get(right_col, pd.Series(index=data.index)), errors="coerce").values

    ax1.plot(x_labels, y1, label=left_label, color=left_color, linewidth=1.8)
    _annotate_last(ax1, x_labels, y1)
    ax1.set_ylabel(left_ylab, color=CSS_COLORS["text"])

    ax2 = ax1.twinx()
    ax2.plot(x_labels, y2, label=right_label, color=right_color, linewidth=1.8, linestyle="--")
    ax2.set_ylabel(right_ylab, color=CSS_COLORS["text"])

    ax1.set_xlabel("Període", color=CSS_COLORS["text"])
    ax1.grid(False)
    _tune_axes(ax1, max_xticks=6, force_all_xticks=force_all_xticks)

    lines, labels = [], []
    for a in (ax1, ax2):
        ln, lb = a.get_legend_handles_labels()
        lines.extend(ln); labels.extend(lb)
    ax1.legend(lines, labels, frameon=False, fontsize=9, loc="upper left", ncol=2)

    return _mpl_finish(fig)

def mpl_dual_bar(df: pd.DataFrame, left_col: str, right_col: str,
                 left_label: str, right_label: str,
                 left_ylab: str, right_ylab: str,
                 start_year: int = SERIES_START_YEAR,
                 left_color: str = GLOBAL_PALETTE["total"],
                 right_color: str = GLOBAL_PALETTE["segunda_ma"],
                 force_all_xticks: bool = True) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax1 = _mpl_base()
    x_labels = _flatten_period_index(data.index)
    x_idx = np.arange(len(x_labels))

    y1 = pd.to_numeric(data.get(left_col, pd.Series(index=data.index)), errors="coerce").values
    y2 = pd.to_numeric(data.get(right_col, pd.Series(index=data.index)), errors="coerce").values

    ax1.bar(x_idx, y1, color=left_color, width=0.6, label=left_label, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax1.set_ylabel(left_ylab, color=CSS_COLORS["text"])
    ax1.set_xticks(x_idx); ax1.set_xticklabels(x_labels)
    _tune_axes(ax1, max_xticks=6, force_all_xticks=force_all_xticks)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, y2, label=right_label, color=right_color, linewidth=1.8, linestyle="--")
    ax2.set_ylabel(right_ylab, color=CSS_COLORS["text"])

    lines, labels = [], []
    for a in (ax1, ax2):
        ln, lb = a.get_legend_handles_labels()
        lines.extend(ln); labels.extend(lb)
    ax1.legend(lines, labels, frameon=False, fontsize=9, loc="upper left", ncol=2)

    return _mpl_finish(fig)

# ========== TABLAS ==========
def _hex_to_rl(hexstr: str):
    return rl_colors.HexColor(hexstr)

def _maybe_flatten_index_and_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out.index = _flatten_period_index(out.index)
    except Exception:
        pass
    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for tpl in out.columns:
            lab = " ".join([str(x) for x in tpl if (x is not None and str(x) != "")])
            new_cols.append(lab)
        out.columns = new_cols
    return out

def _styled_table_from_df(df, max_rows: Optional[int] = None, max_cols: int = 12) -> Table:
    if isinstance(df, str) and "<table" in df.lower():
        try:
            lst = pd.read_html(df)
            if lst: df = lst[0]
        except Exception:
            pass
    if hasattr(df, "data"):
        try:
            df = df.data
        except Exception:
            pass

    df = _maybe_flatten_index_and_cols(pd.DataFrame(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _format_df_thousands(df)


    if max_rows is not None:
        df = df.iloc[:max_rows, :max_cols]
    else:
        df = df.iloc[:, :max_cols]

    data = [[""] + [str(c) for c in df.columns]]
    for idx, row in df.iterrows():
        data.append([str(idx)] + [str(v) for v in row.values])

    tbl = Table(data, repeatRows=1)

    try:
        total_width = 1.15  # 15% más ancha
        tbl._argW = [w * total_width if w else None for w in tbl._argW]
    except Exception:
        pass

    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), _hex_to_rl(CSS_COLORS["primary"])),
        ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), _hex_to_rl(CSS_COLORS["text"])),
        ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.4, _hex_to_rl(CSS_COLORS["bg"])),
        ('LINEBELOW', (0,0), (-1,0), 1, _hex_to_rl(CSS_COLORS["brand_dark"])),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [_hex_to_rl(CSS_COLORS["bg"]), _hex_to_rl(CSS_COLORS["accent"])]),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return tbl

def _header_footer_cover(canvas, doc):
    """
    Cabecera/pie per a la portada: sense logo superior dret ni text de fonts,
    però amb la mateixa franja de color inferior que la resta de pàgines per
    donar continuïtat de marca amb la resta de l'informe.
    """
    canvas.saveState()
    W, H = doc.pagesize
    # Fondo igual que el resto para mantener consistencia
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, W, H, stroke=0, fill=1)
    # Franges de marca (superior i inferior), com a un separador de coberta editorial.
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["primary"]))
    canvas.rect(0, 0, W, 0.35*cm, stroke=0, fill=1)
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["brand_dark"]))
    canvas.rect(0, H - 0.35*cm, W, 0.35*cm, stroke=0, fill=1)
    canvas.restoreState()





def _header_footer_normal(canvas, doc):
    canvas.saveState()

    W, H = doc.pagesize
    margin_x = 1.2*cm
    footer_h = 1.35*cm
    bar_h    = 0.25*cm
    y0       = 0
    y_text   = y0 + bar_h + 0.55*cm

    # Fons
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, W, H, stroke=0, fill=1)

    # Franja inferior
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["primary"]))
    canvas.rect(0, y0, W, bar_h, stroke=0, fill=1)

    # Línia divisòria
    canvas.setStrokeColor(_hex_to_rl(CSS_COLORS["accent"]))
    canvas.setLineWidth(0.6)
    canvas.line(margin_x, y0 + bar_h + 0.35*cm, W - margin_x, y0 + bar_h + 0.35*cm)

    # Colors/textos
    txt_color  = _hex_to_rl(CSS_COLORS["text"])
    link_color = _hex_to_rl(CSS_COLORS["primary"])

    # Esquerra: fonts
    canvas.setFillColor(txt_color)
    canvas.setFont("Helvetica-Oblique", 9)
    left_text = "Font de les dades: APCE, Agència de l'Habitatge de Catalunya, INCASÒL, INE."
    canvas.drawString(margin_x, y_text, left_text)

    # Centre: web clicable
    center_text = "www.apcebcn.cat"
    canvas.setFont("Helvetica-Bold", 9)
    tw_center = canvas.stringWidth(center_text, "Helvetica-Bold", 9)
    cx = W/2 - tw_center/2
    canvas.setFillColor(link_color)
    canvas.drawString(cx, y_text, center_text)
    try:
        canvas.linkURL("https://apcebcn.cat/", (cx, y_text-1, cx+tw_center, y_text+10), relative=0, thickness=0)
    except Exception:
        pass

    # Dreta: badge només amb número de pàgina, mida i amplada adaptativa
    page_text = str(doc.page)
    font_name = "Helvetica-Bold"
    font_size = 13  # número més gran
    canvas.setFont(font_name, font_size)

    # Amplada del text (punts)
    tw = canvas.stringWidth(page_text, font_name, font_size)

    # Padding en punts (adaptatius)
    pad_x = 8   # ~2.8 mm
    pad_y = 4   # ~1.4 mm

    badge_w = tw + 2*pad_x
    badge_h = font_size + 2*pad_y  # suficient per encabir l’altura del text

    bx = W - margin_x - badge_w
    # Vertical: alineem amb la línia base del text central
    by = y_text - (badge_h - font_size)/2 - 1

    # Pastilla arrodonida
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["accent"]))
    try:
        canvas.roundRect(bx, by, badge_w, badge_h, 6, stroke=0, fill=1)
    except Exception:
        canvas.rect(bx, by, badge_w, badge_h, stroke=0, fill=1)

    # Número centrat
    canvas.setFillColor(txt_color)
    tx = bx + (badge_w - tw)/2
    ty = by + (badge_h - font_size)/2 - 1  # petit ajust òptic
    canvas.drawString(tx, ty, page_text)

    canvas.restoreState()


def _header_footer_minimal(canvas, doc):
    # Solo fondo, sin logo ni número ni fuente
    canvas.saveState()
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
    canvas.restoreState()

def build_location_pdf_ordered(
    location_name: str,
    kpis: List[Tuple[str, str, Optional[str]]],
    sections: List[Tuple[str, List[Tuple[str, object]]]],  # [(titulo_seccion, [("table", (titulo, df)) o ("fig", (titulo, png_bytes)) , ...])]
) -> bytes:
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=PDF_PAGE_SIZE,
        rightMargin=1.75 * cm,
        leftMargin=1.75 * cm,
        topMargin=2.5 * cm,
        bottomMargin=1.75 * cm
    )

    # === Crear el frame común ===
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='frame')

    # === Añadir las tres plantillas de página ===
    tpl_cover   = PageTemplate(id='Cover',   frames=frame, onPage=_header_footer_cover)
    tpl_normal  = PageTemplate(id='Normal',  frames=frame, onPage=_header_footer_normal)
    tpl_minimal = PageTemplate(id='Minimal', frames=frame, onPage=_header_footer_minimal)
    doc.addPageTemplates([tpl_cover, tpl_normal, tpl_minimal])
    doc.title = f"Informe de mercat residencial (APCE) — {location_name}"
    doc.author = "APCE CATALUNYA"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBrand", parent=styles["Title"], fontSize=18,
                              textColor=_hex_to_rl(CSS_COLORS["brand_dark"])))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=14,
                              textColor=_hex_to_rl(CSS_COLORS["brand_dark"]), spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9,
                              textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(name="KPI", parent=styles["BodyText"], fontSize=14, leading=17,
                              textColor=_hex_to_rl(CSS_COLORS["text"])))
    # Targeta de KPI en dues línies (etiqueta petita a sobre, valor destacat a sota),
    # centrada, per a la graella de KPIs de 3 columnes de la portada de dades.
    styles.add(ParagraphStyle(name="KPICard", parent=styles["BodyText"], fontSize=11,
                              leading=14, alignment=1, spaceBefore=0, spaceAfter=0,
                              textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(
        name="SectionBand", parent=styles["Heading2"], fontSize=14, leading=16,
        textColor=_hex_to_rl("#ffffff"), backColor=_hex_to_rl(CSS_COLORS["primary"]),
        leftIndent=0, rightIndent=0, spaceBefore=8, spaceAfter=6, alignment=0
    ))
    story = []



    def append_cover_page(story, styles, location_name, logo_path=LOGO_APCE):
        story.append(Spacer(1, 2 * cm))  # margen superior

        # Logo grande centrado
        try:
            logo = RLImage(logo_path, width=10 * cm, height=5 * cm)
            logo.hAlign = 'CENTER'
            story.append(logo)
        except Exception:
            story.append(Spacer(1, 6 * cm))

        story.append(Spacer(1, 1.0 * cm))

        # Título principal
        story.append(Paragraph(
            "INFORME DE MERCAT RESIDENCIAL",
            ParagraphStyle(
                "CoverTitle",
                parent=styles["Title"],
                fontSize=28,
                leading=32,
                alignment=1,  # centrado
                textColor=_hex_to_rl(CSS_COLORS["brand_dark"]),
                spaceAfter=12
            )
        ))

        # Subtítulo con nombre del municipio
        story.append(Paragraph(
            f"<b>{location_name.upper()}</b>",
            ParagraphStyle(
                "CoverSub",
                parent=styles["BodyText"],
                fontSize=20,
                alignment=1,
                textColor=_hex_to_rl(CSS_COLORS["primary"]),
                spaceAfter=20
            )
        ))

        # Línea divisoria fina (opcional)
        story.append(Spacer(1, 0.3 * cm))
        tbl = Table([[""]], colWidths=[16 * cm], rowHeights=[0.05 * cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), _hex_to_rl("#e0e0e0")),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.8 * cm))

        # Texto de autoría
        story.append(Paragraph(
            "Elaborat per <b>APCE Catalunya</b>",
            ParagraphStyle(
                "CoverMeta",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=12,
                textColor=_hex_to_rl(CSS_COLORS["text"]),
                spaceAfter=6
            )
        ))

        # Fecha de generación
        story.append(Paragraph(
            f"Data de generació de l'informe: {datetime.now():%d/%m/%Y}",
            ParagraphStyle(
                "CoverMeta2",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=11,
                textColor=_hex_to_rl("#777777")
            )
        ))

        # Empujar hacia el final
        story.append(Spacer(1, 0.5 * cm))

        # Enlace institucional (clicable)
        story.append(Paragraph(
            f'<link href="https://apcebcn.cat/es/" color="{CSS_COLORS["primary"]}">www.apcebcn.cat</link>',
            ParagraphStyle(
                "CoverLink",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=12,
                textColor=_hex_to_rl(CSS_COLORS["primary"])
            )
        ))
        # ⬇⬇⬇ AÑADIR ESTAS DOS LÍNEAS ANTES DEL SALTO ⬇⬇⬇

        story.append(NextPageTemplate('Normal'))
        story.append(PageBreak())







    # === estilos extra para la página final ===
    styles.add(ParagraphStyle(name="CenterBig", parent=styles["BodyText"],
                            alignment=1, fontSize=14,
                            textColor=_hex_to_rl(CSS_COLORS["brand_dark"])))
    styles.add(ParagraphStyle(name="Center", parent=styles["BodyText"],
                            alignment=1, fontSize=11,
                            textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(name="SmallCorner", parent=styles["BodyText"],
                            alignment=2, fontSize=8,
                            textColor=_hex_to_rl("#777777")))

    def append_closing_page(story, styles, logo_path=LOGO_CLOSING):
        # Mida reduïda (mantenint la ràtio 16:9 original de la imatge) perquè, juntament
        # amb la llegenda del període, càpiga còmodament en l'alçada disponible de la
        # pàgina panoràmica (PDF_PAGE_SIZE), més baixa que l'antic A4 apaïsat.
        closing_block = []
        try:
            logo = RLImage(logo_path, width=22*cm, height=12.375*cm)
            logo.hAlign = 'CENTER'
            closing_block.append(Spacer(1, 0*cm))
            closing_block.append(logo)
        except Exception:
            closing_block.append(Spacer(1, 9.0*cm))
        # Període de referència de l'Estudi d'Oferta (Atlas) mostrat a la darrera pàgina.
        closing_block.append(Spacer(1, 0.3*cm))
        closing_block.append(Paragraph(f"H1_{CURRENT_YEAR_LIMIT}", styles["SmallCorner"]))
        story.append(KeepTogether(closing_block))


    # === Portada ===
    append_cover_page(story, styles, location_name=location_name, logo_path=LOGO_APCE)


    story.append(Paragraph(
        f"Informe de mercat residencial (APCE): Municipi de {location_name}",
        styles["TitleBrand"]
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Darreres dades dels indicadors socioeconòmics i del mercat residencial a {location_name}", styles["Section"]))

    if kpis and len(kpis) > 0:
        # Cada KPI es mostra com una petita targeta de dues línies: etiqueta a sobre
        # (petita, en majúscules) i valor destacat a sota (amb la variació de color si n'hi ha).
        kpi_paragraphs = []
        for label, val, delta in kpis:
            label_html = f'<font size="9" color="{CSS_COLORS["brand_dark"]}"><b>{label.upper()}</b></font>'
            value_html = f'<font size="15" color="{CSS_COLORS["text"]}"><b>{val}</b></font>'
            delta_html = (" " + _delta_fmt(delta)) if delta else ""
            kpi_paragraphs.append(Paragraph(f"{label_html}<br/>{value_html}{delta_html}", styles["KPICard"]))

        # Graella de 3 columnes (millor aprofitament de l'amplada panoràmica que les
        # antigues 2 columnes), omplerta per files en l'ordre natural de lectura.
        n = len(kpi_paragraphs)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        while len(kpi_paragraphs) < nrows * ncols:
            kpi_paragraphs.append(Paragraph("", styles["KPICard"]))
        kpi_data = [kpi_paragraphs[r*ncols:(r+1)*ncols] for r in range(nrows)]

        kpi_tbl = Table(kpi_data, colWidths=[doc.width/ncols]*ncols, rowHeights=1.35*cm)
        kpi_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), _hex_to_rl(CSS_COLORS["accent"])),
            ('BOX', (0,0), (-1,-1), 0.5, _hex_to_rl(CSS_COLORS["bg"])),
            ('INNERGRID', (0,0), (-1,-1), 2, _hex_to_rl(CSS_COLORS["bg"])),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(kpi_tbl)

    else:
        story.append(Paragraph("No hi ha KPIs disponibles.", styles["Small"]))

    story.append(Spacer(1, 0.1*cm))
    story.append(PageBreak())
    # ====== SECCIONES con salto de página entre ellas ======
    for si, (section_title, items) in enumerate(sections):
        # Banda de color amb el nom del bloc temàtic (Producció, Compravendes, Preus...):
        # dona una jerarquia visual clara entre blocs, similar a un separador de capítol.
        band = Paragraph(section_title, styles["SectionBand"])
        band.keepWithNext = True
        story.append(band)
        story.append(Spacer(1, TITLE_SPACING_CM*cm))

        for kind, payload in items:
            if kind == "table":
                title, df = payload
                hdr = Paragraph(title, styles["Section"])
                hdr.keepWithNext = True
                story.append(hdr)
                try:
                    df_disp = df.data if hasattr(df, "data") else df
                    tbl = _styled_table_from_df(df_disp, max_rows=None, max_cols=12)
                    story.append(tbl)
                except Exception:
                    story.append(Paragraph("[No s'ha pogut mostrar la taula]", styles["Small"]))
                story.append(Spacer(1, 1*cm))

            elif kind == "fig":
                title, png_bytes = payload
                hdr = Paragraph(title, styles["Section"])
                hdr.keepWithNext = True
                img = Image(io.BytesIO(png_bytes), width=23*cm, height=11*cm)
                story.append(KeepTogether([hdr, img]))
                story.append(Spacer(1, 1*cm))

        if si < len(sections) - 1:
            story.append(Spacer(1, 0.5*cm))
            story.append(CondPageBreak(8*cm))  # rompe si quedan < 8 cm libres
    story.append(NextPageTemplate('Minimal'))
    append_closing_page(story, styles, logo_path=LOGO_CLOSING)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ========== HELPERS EXTRA (ALTRES INDICADORS) ==========
def mpl_donut(labels, values) -> bytes:
    fig, ax = _mpl_base()
    donut_colors = ["#2d538f", "#D9773F", "#1b7f3a", "#6a3d9a", "#b15928", "#727375"][:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, startangle=90, colors=donut_colors,
        wedgeprops=dict(width=0.45, edgecolor=CSS_COLORS["bg"]),
        autopct="%1.1f%%", pctdistance=0.75
    )
    for t in texts:
        t.set_fontsize(9)
        t.set_color(CSS_COLORS["text"])
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
        at.set_weight("bold")
    ax.axis("equal")
    ax.grid(False)
    return _mpl_finish(fig)

def _map_df_mun_idescat_basic(df_mun_idescat: pd.DataFrame, selected_mun: str) -> Optional[pd.DataFrame]:
    """
    Devuelve un DataFrame con una columna adicional 'nombre_largo'
    que mapea las variables internas (de df_mun_idescat) a sus nombres
    descriptivos en catalán, eliminando el sufijo del municipio.

    Se usa para extraer valores agregados (last_year, last_value) de
    indicadores demográficos, económicos o laborales.
    """

    if df_mun_idescat is None or "variable" not in df_mun_idescat.columns:
        return None

    # Diccionario maestro de variables reconocidas
    name_map = {
        # ECONOMIA I RENDA
        "IRPF_Base_imposable": "Base imposable mitjana de l’IRPF (€)",
        "Pensionistes_Total": "Nombre de pensionistes",
        "Parc_vehicles_Total": "Parc total de vehicles",
        "Residus_mun_per_capita": "Residus municipals per càpita (kg/hab/dia)",

        # MERCAT LABORAL
        "AfiliatSS_Agricultura": "Afiliats a la Seguretat Social – Agricultura",
        "AfiliatSS_Construcció": "Afiliats a la Seguretat Social – Construcció",
        "AfiliatSS_Indústria": "Afiliats a la Seguretat Social – Indústria",
        "AfiliatSS_Serveis": "Afiliats a la Seguretat Social – Serveis",
        "AfiliatSS_Total": "Afiliats a la Seguretat Social – Total",
        "Atur registrat_Agricultura": "Atur registrat – Agricultura",
        "Atur registrat_Construcció": "Atur registrat – Construcció",
        "Atur registrat_Indústria": "Atur registrat – Indústria",
        "Atur registrat_Serveis": "Atur registrat – Serveis",
        "Atur registrat_Total": "Atur registrat – Total",
        "poblacio_activa": "Població activa",
        "poblacio_ocupada": "Població ocupada",
        "poblacio_desocupada": "Població desocupada",
        "poblacio_inactiva": "Població inactiva",

        # DEMOGRAFIA
        "Matrimonis_Total": "Nombre de matrimonis",
        "Naixements_Total": "Nombre de naixements",
    }

    df = df_mun_idescat.copy()
    df["variable_sin_municipi"] = df["variable"].astype(str).str.replace(
        f"_{selected_mun}$", "", regex=True
    )
    df["nombre_largo"] = df["variable_sin_municipi"].map(name_map)
    return df


def _pick_last_val(df_long: pd.DataFrame, long_label: str):
    try:
        row = df_long.loc[df_long["nombre_largo"] == long_label].iloc[0]
        return int(row["last_year"]), float(row["last_value"])
    except Exception:
        return None, None

# ========== GENERADOR — MUNICIPI (ORDEN COHERENTE) ==========
def generar_pdf_municipi_tot(
    selected_mun: str,
    # --- Producció
    table_mun_prod: pd.DataFrame, table_mun_prod_y: pd.DataFrame,
    table_mun_prod_pluri: pd.DataFrame, table_mun_prod_uni: pd.DataFrame,
    selected_columns_ini: List[str], selected_columns_fin: List[str],
    # --- Compravendes
    table_mun_tr: pd.DataFrame, table_mun_tr_y: pd.DataFrame,
    # --- Preus
    table_mun_pr: pd.DataFrame, table_mun_pr_y: pd.DataFrame,
    # --- Superfície
    table_mun_sup: pd.DataFrame, table_mun_sup_y: pd.DataFrame,
    # --- Lloguer
    table_mun_llog: pd.DataFrame, table_mun_llog_y: pd.DataFrame,
    # --- Altres indicadors (dataframes globales ya cargados)
    censo_2021=None, DT_mun_y=None, idescat_muns=None, rentaneta_mun=None, tabla_estudi_oferta=None
):
    """Genera el PDF del municipi con secciones ordenadas (tabla(s) → gráfico(s)) y salto de página entre indicadores."""
    # ==========================
    # 1) KPIs
    # ==========================
    kpis_pdf = []

    def _safe_add_kpi(table_y, table_q, col, label):
        try:
            year = str(CURRENT_YEAR_LIMIT)
            val = indicator_year(table_y, table_q, year, col, "level")
            var = indicator_year(table_y, table_q, year, col, "var")
        except Exception:
            try:
                last_year = str(table_y.index[-1])
                val = indicator_year(table_y, table_q, last_year, col, "level")
                var = indicator_year(table_y, table_q, last_year, col, "var")
            except Exception:
                kpis_pdf.append((label, "No disponible", None))
                return
        kpis_pdf.append((label, f"{val:,.0f}".replace(",", "."), f"{var}%"))

    # Producció — totals + tipologies
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats", "Habitatges iniciats")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats", "Habitatges acabats")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats plurifamiliars", "Iniciats plurifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats unifamiliars", "Iniciats unifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats plurifamiliars", "Acabats plurifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats unifamiliars", "Acabats unifamiliars")
    try:
        if DT_mun_y is not None:
            col_prov = f"calprovgene_{selected_mun}"
            col_def  = f"caldefgene_{selected_mun}"
            cols_ok = [c for c in [col_prov, col_def] if c in DT_mun_y.columns]
            if cols_ok:
                # Base limpia (desde 2000)
                df_vpo = (
                    DT_mun_y.loc[:, ["Fecha"] + cols_ok]
                    .dropna(how="all", subset=cols_ok)
                    .assign(Fecha=lambda d: pd.to_numeric(d["Fecha"], errors="coerce").astype("Int64"))
                    .dropna(subset=["Fecha"])
                    .assign(Fecha=lambda d: d["Fecha"].astype(int))
                    .sort_values("Fecha")
                    .drop_duplicates(subset=["Fecha"], keep="last")
                )
                df_vpo = df_vpo[df_vpo["Fecha"] >= 2000]

                for col, label in [
                    (col_prov, "Qualificacions provisionals d'HPO"),
                    (col_def,  "Qualificacions definitives d'HPO"),
                ]:
                    if col in df_vpo.columns and not df_vpo[col].dropna().empty:
                        df_col = df_vpo.dropna(subset=[col])
                        last_year = int(df_col["Fecha"].iloc[-1])
                        last_val  = float(df_col[col].iloc[-1])
                        # delta vs. año anterior (si existe y no es 0)
                        prev = df_col.loc[df_col["Fecha"] == last_year - 1, col]
                        delta = None
                        if not prev.empty and float(prev.iloc[0]) != 0:
                            delta = f"{(100.0 * (last_val / float(prev.iloc[0]) - 1)):.1f}%"

                        kpis_pdf.append((
                            f"{label} ({last_year})",
                            f"{last_val:,.0f}".replace(",", "."),
                            delta
                        ))
    except Exception:
        pass

    # Compravendes
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge total", "Compravendes")
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge de segona mà", "Compravendes segona mà")
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge nou", "Compravendes habitatge nou")

    # Preus
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge total", "Preu €/m²")
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge de segona mà", "Preu €/m² segona mà")
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge nou", "Preu €/m² nou")

    # Superfície
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana total", "Superfície mitjana (m² construït)")
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana segona mà (m² construït)")
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana d'habitatge nou", "Superfície mitjana nou (m² construït)")

    # Lloguer
    _safe_add_kpi(table_mun_llog_y, table_mun_llog, "Nombre de contractes de lloguer", "Contractes de lloguer")
    _safe_add_kpi(table_mun_llog_y, table_mun_llog, "Rendes mitjanes de lloguer", "Renda mitjana lloguer (€/mes)")
    # === Altres indicadors -> IRPF al bloque de KPIs ===
    # === BLOQUE UNIFICADO: Altres indicadors (df_mun_idescat) ===
    try:
        if censo_2021 is not None:
            row = censo_2021[censo_2021["Municipi"] == selected_mun].iloc[0]

            # evita duplicados si este bloque se ejecuta más de una vez
            parc_labels = {
                "Propietat", "Habitatges principals", "Habitatges no principals",
                "Habitatges en lloguer", "Edat mitjana habitatges", "Superfície mitjana (m²)"
            }
            kpis_pdf = [k for k in kpis_pdf if k[0] not in parc_labels]

            parc_kpis = [
                ("Habitatges en propietat",                f"{float(row['Perc_propiedad']):.1f}%", None),
                ("Habitatges principals",    f"{(100.0 - float(row['Perc_noprincipales_y'])):.1f}%", None),
                ("Habitatges no principals", f"{float(row['Perc_noprincipales_y']):.1f}%", None),
                ("Habitatges en lloguer",    f"{float(row['Perc_alquiler']):.1f}%", None),
                ("Edat mitjana habitatges",  f"{float(row['Edad media']):.1f}", None),
                ("Superfície mitjana (m²)",  f"{float(row['Superficie media']):.1f}", None),
            ]

            # Añadirlos al final del listado de KPIs
            kpis_pdf.extend(parc_kpis)
    except Exception:
        pass
    try:
        df_long = _map_df_mun_idescat_basic(df_mun_idescat, selected_mun)
        if df_long is not None and not df_long.empty:

            def _append_if_ok(nombre_largo: str, label_fmt: Optional[str] = None, fmt: str = "int"):
                yr, val = _pick_last_val(df_long, nombre_largo)
                if yr is not None and pd.notnull(val):
                    label = label_fmt.format(yr) if label_fmt else f"{nombre_largo} ({yr})"
                    if fmt == "int":
                        val_str = f"{int(round(val)):,}".replace(",", ".")
                    elif fmt == "float1":
                        val_str = f"{float(val):.1f}"
                    elif fmt == "float2":
                        val_str = f"{float(val):.2f}"
                    else:
                        val_str = str(val)
                    kpis_pdf.append((label, val_str, None))

            # --- MERCAT LABORAL ---
            _append_if_ok("Afiliats a la Seguretat Social – Agricultura")
            _append_if_ok("Afiliats a la Seguretat Social – Construcció")
            _append_if_ok("Afiliats a la Seguretat Social – Indústria")
            _append_if_ok("Afiliats a la Seguretat Social – Serveis")
            _append_if_ok("Afiliats a la Seguretat Social – Total")

            _append_if_ok("Atur registrat – Agricultura")
            _append_if_ok("Atur registrat – Construcció")
            _append_if_ok("Atur registrat – Indústria")
            _append_if_ok("Atur registrat – Serveis")
            _append_if_ok("Atur registrat – Total")

            _append_if_ok("Població activa")
            _append_if_ok("Població ocupada")
            _append_if_ok("Població desocupada")
            _append_if_ok("Població inactiva")
            # --- ECONOMIA I RENDA ---
            _append_if_ok("Base imposable mitjana de l’IRPF (€)")
            _append_if_ok("Nombre de pensionistes")
            _append_if_ok("Parc total de vehicles")
            _append_if_ok("Residus municipals per càpita (kg/hab/dia)", fmt="float2")

            # --- DEMOGRAFIA ---
            _append_if_ok("Nombre de matrimonis")
            _append_if_ok("Nombre de naixements")

    except Exception as e:
        print(f"⚠️ Error al afegir Altres indicadors al bloc de KPIs: {e}")
        pass



    # ==========================
    # 2) SECCIONES (tabla(s) → gráfico(s))
    # ==========================
    sections: List[Tuple[str, List[Tuple[str, Tuple[str, object]]]]] = []

    # --------- PRODUCCIÓ ---------
    items_produccio = []
    try:
        items_produccio.append((
            "table",
            (f"Evolució trimestral de la producció d'habitatges al municipi de {selected_mun}",
             table_trim(table_mun_prod, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "table",
            (f"Evolució anual de la producció d'habitatges al municipi de {selected_mun}",
             table_year(table_mun_prod_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Evolució trimestral dels habitatges iniciats i acabats al municipi de {selected_mun}",
             mpl_line(table_mun_prod, ["Habitatges iniciats", "Habitatges acabats"], "", "Habitatges",
                      start_year=SERIES_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Evolució anual dels habitatges iniciats i acabats al municipi de {selected_mun}",
             mpl_bar(table_mun_prod_y, ["Habitatges iniciats", "Habitatges acabats"], "", "Habitatges",
                     start_year=SERIES_START_YEAR, force_all_xticks=True))
        ))
    except Exception:
        pass
    # Tipologies
    try:
        typ_ini_cols = selected_columns_ini
        typ_ini_palette = [GLOBAL_PALETTE["unifamiliar"] if "unifam" in c.lower()
                           else GLOBAL_PALETTE["plurifamiliar"] if "plurifam" in c.lower()
                           else GLOBAL_PALETTE["total"] for c in typ_ini_cols]
        items_produccio.append((
            "fig",
            (f"Evolució dels habitatges iniciats per tipologia al municipi de {selected_mun}",
             mpl_area(table_mun_prod[typ_ini_cols], typ_ini_cols, "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR, palette=typ_ini_palette))
        ))
    except Exception:
        pass

    try:
        typ_fin_cols = selected_columns_fin
        typ_fin_palette = [GLOBAL_PALETTE["unifamiliar"] if "unifam" in c.lower()
                           else GLOBAL_PALETTE["plurifamiliar"] if "plurifam" in c.lower()
                           else GLOBAL_PALETTE["total"] for c in typ_fin_cols]
        items_produccio.append((
            "fig",
            (f"Evolució dels habitatges acabats per tipologia al municipi de {selected_mun}",
             mpl_area(table_mun_prod[typ_fin_cols], typ_fin_cols, "", "Habitatges acabats",
                      start_year=SERIES_START_YEAR, palette=typ_fin_palette))
        ))
    except Exception:
        pass

    # Per superfície
    try:
        items_produccio.append((
            "fig",
            (f"Habitatges iniciats plurifamiliars per superfície al municipi de {selected_mun}",
             mpl_area(table_mun_prod_pluri, table_mun_prod_pluri.columns.tolist(), "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR,
                      palette=["#2d538f", "#1b7f3a", "#D9773F", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Habitatges iniciats unifamiliars per superfície al municipi de {selected_mun}",
             mpl_area(table_mun_prod_uni, table_mun_prod_uni.columns.tolist(), "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR,
                      palette=["#2d538f", "#1b7f3a", "#D9773F", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]))
        ))
    except Exception:
        pass
    if items_produccio:
        sections.append(("Producció", items_produccio))
# --------- HABITATGE PROTEGIT (HPO) ---------
    items_vpo = []
    try:
        if DT_mun_y is not None:
            col_prov = f"calprovgene_{selected_mun}"
            col_def  = f"caldefgene_{selected_mun}"
            cols_ok = [c for c in [col_prov, col_def] if c in DT_mun_y.columns]
            if cols_ok:
                df_vpo = DT_mun_y.loc[:, ["Fecha"] + cols_ok].dropna(how="all", subset=cols_ok).copy()
                df_vpo["Fecha"] = pd.to_numeric(df_vpo["Fecha"], errors="coerce").astype("Int64")
                df_vpo = df_vpo.dropna(subset=["Fecha"]).copy()
                df_vpo["Fecha"] = df_vpo["Fecha"].astype(int)
                df_vpo = df_vpo.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
                df_vpo = df_vpo[df_vpo["Fecha"] >= 2000].copy()  # desde 2000
                df_vpo = df_vpo.rename(columns={
                    col_prov: "Qualificacions provisionals HPO",
                    col_def:  "Qualificacions definitives HPO"
                })
                df_vpo = df_vpo.set_index("Fecha")

                # Tabla (transpuesta para encajar con tu estilo)
                items_vpo.append((
                    "table",
                    (f"Evolució de les qualificacions anuals d'habitatge protegit (HPO) al municipi de {selected_mun}",
                    df_vpo[df_vpo.index>2012].T)
                ))

                # Gráfico de líneas (desde 2000)
                items_vpo.append((
                    "fig",
                    (f"",
                    mpl_line(
                        df_vpo,
                        [c for c in ["Qualificacions provisionals HPO", "Qualificacions definitives HPO"] if c in df_vpo.columns],
                        title="",
                        ylab="Habitatges",
                        xlab="Any",
                        start_year=2000,
                        force_all_xticks=True
                    ))
                ))
    except Exception:
        pass

    if items_vpo:
        sections.append(("Habitatge protegit (HPO)", items_vpo))

    # --------- COMPRAVENDES ---------
    items_comp = []
    try:
        items_comp.append((
            "table",
            (f"Evolució trimestral de les compravendes al municipi de {selected_mun}",
             table_trim(table_mun_tr, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_comp.append((
            "table",
            (f"Evolució anual de les compravendes al municipi de  {selected_mun}",
             table_year(table_mun_tr_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    comp_cols = [
        "Compravendes d'habitatge total",
        "Compravendes d'habitatge de segona mà",
        "Compravendes d'habitatge nou"
    ]
    comp_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_comp.append((
            "fig",
            (f"Evolució trimestral de les compravendes al municipi de {selected_mun}",
             mpl_line(table_mun_tr, comp_cols, "", "Operacions",
                      start_year=SERIES_START_YEAR, palette=comp_palette))
        ))
    except Exception:
        pass
    try:
        items_comp.append((
            "fig",
            (f"Evolució anual de les compravendes al municipi de {selected_mun}",
             mpl_bar(table_mun_tr_y, comp_cols, "", "Operacions",
                     start_year=SERIES_START_YEAR, palette=comp_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_comp:
        sections.append(("Compravendes", items_comp))

    # --------- PREUS ---------
    items_preus = []
    try:
        items_preus.append((
            "table",
            (f"Evolució trimestral dels preus al municipi de {selected_mun}",
             table_trim(table_mun_pr, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_preus.append((
            "table",
            (f"Evolució anual dels preus €/m² al municipi de {selected_mun}",
             table_year(table_mun_pr_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    preus_cols = [
        "Preu d'habitatge total",
        "Preu d'habitatge de segona mà",
        "Preu d'habitatge nou"
    ]
    preus_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_preus.append((
            "fig",
            (f"Evolució trimestral dels preus €/m² al municipi de {selected_mun}",
             mpl_line(table_mun_pr, preus_cols, "", "€/m²",
                      start_year=SERIES_START_YEAR, palette=preus_palette))
        ))
    except Exception:
        pass
    try:
        items_preus.append((
            "fig",
            (f"Evolució anual dels preus €/m² al municipi de {selected_mun}",
             mpl_bar(table_mun_pr_y, preus_cols, "", "€/m²",
                     start_year=SERIES_START_YEAR, palette=preus_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_preus:
        sections.append(("Preus", items_preus))

    # --------- SUPERFÍCIE ---------
    items_sup = []
    try:
        items_sup.append((
            "table",
            (f"Evolució trimestral de la superfície en m² construïts al municipi de {selected_mun}",
             table_trim(table_mun_sup, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_sup.append((
            "table",
            (f"Evolució anual de la superfície en m² construïts al municipi de {selected_mun}",
             table_year(table_mun_sup_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    sup_cols = [
        "Superfície mitjana total",
        "Superfície mitjana d'habitatge de segona mà",
        "Superfície mitjana d'habitatge nou"
    ]
    sup_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_sup.append((
            "fig",
            (f"Evolució trimestral de la superfície en m² construïts al municipi de {selected_mun}",
             mpl_line(table_mun_sup, sup_cols, "", "m²",
                      start_year=SERIES_START_YEAR, palette=sup_palette))
        ))
    except Exception:
        pass
    try:
        items_sup.append((
            "fig",
            (f"Evolució anual de la superfície en m² construïts al municipi de {selected_mun}",
             mpl_bar(table_mun_sup_y, sup_cols, "", "m²",
                     start_year=SERIES_START_YEAR, palette=sup_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_sup:
        sections.append(("Superfície", items_sup))

    # --------- LLOGUER ---------
    items_llog = []
    try:
        items_llog.append((
            "table",
            (f"Evolució trimestral del mercat de lloguer al municipi de {selected_mun}",
             table_trim(table_mun_llog, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_llog.append((
            "table",
            (f"Evolució anual del mercat de lloguer al municipi de {selected_mun}",
             table_year(table_mun_llog_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    # Doble eje (trimestral)
    if ("Nombre de contractes de lloguer" in getattr(table_mun_llog, "columns", [])) and \
       ("Rendes mitjanes de lloguer" in getattr(table_mun_llog, "columns", [])):
        try:
            items_llog.append((
                "fig",
                (f"Evolució del mercat de lloguer al municipi de {selected_mun}",
                 mpl_dual_line(table_mun_llog,
                               left_col="Nombre de contractes de lloguer",
                               right_col="Rendes mitjanes de lloguer",
                               left_label="Contractes", right_label="Renda mitjana",
                               left_ylab="Contractes", right_ylab="€ / mes",
                               start_year=SERIES_START_YEAR,
                               left_color=GLOBAL_PALETTE["total"],
                               right_color=GLOBAL_PALETTE["segunda_ma"],
                               force_all_xticks=False))
            ))
        except Exception:
            pass
    # Doble eje (anual)
    if ("Nombre de contractes de lloguer" in getattr(table_mun_llog_y, "columns", [])) and \
       ("Rendes mitjanes de lloguer" in getattr(table_mun_llog_y, "columns", [])):
        try:
            items_llog.append((
                "fig",
                (f"Evolució del mercat de lloguer al municipi de {selected_mun}",
                 mpl_dual_bar(table_mun_llog_y,
                              left_col="Nombre de contractes de lloguer",
                              right_col="Rendes mitjanes de lloguer",
                              left_label="Contractes", right_label="Renda mitjana",
                              left_ylab="Contractes", right_ylab="€ / mes",
                              start_year=SERIES_START_YEAR,
                              left_color=GLOBAL_PALETTE["total"],
                              right_color=GLOBAL_PALETTE["segunda_ma"],
                              force_all_xticks=True))
            ))
        except Exception:
            pass
    if items_llog:
        sections.append(("Lloguer", items_llog))

    # --------- DEMOGRAFIA: Població ---------
    items_demo_pop = []
    try:
        pop_col = f"poptottine_{selected_mun}"
        if DT_mun_y is not None and pop_col in DT_mun_y.columns:
            df_pop = DT_mun_y.loc[:, ["Fecha", pop_col]].dropna().copy()
            df_pop["Fecha"] = pd.to_numeric(df_pop["Fecha"], errors="coerce").astype("Int64")
            df_pop = df_pop.dropna(subset=["Fecha"]).copy()
            df_pop["Fecha"] = df_pop["Fecha"].astype(int)
            df_pop = df_pop.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
            df_pop = df_pop.set_index("Fecha")
            df_pop[pop_col] = pd.to_numeric(df_pop[pop_col], errors="coerce")
            df_pop = df_pop[df_pop.index >= 2000]
            df_pop = df_pop.rename(columns={pop_col: "Població"})

            # KPI población
            try:
                last_year = int(df_pop.index[-1])
                last_val = int(df_pop["Població"].iloc[-1])
                prev_year = last_year - 5
                if prev_year in df_pop.index:
                    prev_val = float(df_pop.loc[prev_year, "Població"])
                    delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                else:
                    delta = None
                    if len(df_pop) >= 6:
                        prev_val = float(df_pop["Població"].iloc[-6])
                        delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                kpis_pdf.append(("Població (últim any)", f"{last_val:,.0f}".replace(",", "."), delta))
            except Exception:
                pass

            # Tabla (transpuesta)
            items_demo_pop.append((
                "table",
                (f"Evolució anual de la població al municipi de {selected_mun}", df_pop[df_pop.index>=2015].T)
            ))
            # Gráfico línea
            items_demo_pop.append((
                "fig",
                (f"",
                 mpl_line(df_pop, ["Població"], title="", ylab="Persones", xlab="Any",
                          start_year=2000, force_all_xticks=True))
            ))
    except Exception:
        pass
    if items_demo_pop:
        sections.append(("Demografia — Població", items_demo_pop))

    # --------- DEMOGRAFIA: Tamaño de llar (Censo 2021) ---------
    items_demo_llar = []
    try:
        if censo_2021 is not None:
            row = censo_2021[censo_2021["Municipi"] == selected_mun].iloc[0]
            labels = ["1", "2", "3", "4", "5 o más"]
            vals = [row.get("1", 0), row.get("2", 0), row.get("3", 0), row.get("4", 0), row.get("5 o más", 0)]
            items_demo_llar.append((
                "fig",
                (f"Distribució per grandària de llar al municipi de {selected_mun} (Cens 2021)",
                 mpl_donut(labels, vals))
            ))
            # KPIs adicionales
            try:
                kpis_pdf.append(("Grandària de la llar més freqüent", f"{row['Tamaño_hogar_frecuente']} llars", None))
                kpis_pdf.append(("Grandària mitjà de la llar", f"{float(row['Tamaño medio del hogar']):.2f}", None))
                kpis_pdf.append(("Població nacional", f"{(100.0 - float(row['Perc_extranjera'])):.1f}%", None))
                kpis_pdf.append(("Població estrangera", f"{float(row['Perc_extranjera']):.1f}%", None))
            except Exception:
                pass
    except Exception:
        pass
    if items_demo_llar:
        sections.append(("Demografia — Llar", items_demo_llar))

    # --------- ECONOMIA: Renda neta per llar ---------
    items_renda = []
    try:
        if rentaneta_mun is not None:
            df_rn = rentaneta_mun.rename(columns={"Año": "Any"}).copy()
            col_rn = f"rentanetahogar_{selected_mun}"
            if col_rn in df_rn.columns:
                df_rn = df_rn[["Any", col_rn]].dropna().rename(columns={col_rn: "Renda neta per llar"})
                df_rn = df_rn.set_index("Any")
                try:
                    any_rn = int(df_rn.index[-1])
                    val_rn = float(df_rn.iloc[-1, 0])
                    kpis_pdf.append((f"Renda neta per llar ({any_rn})", f"{val_rn:,.0f}".replace(",", "."), None))
                except Exception:
                    pass

                # Tabla (transpuesta)
                items_renda.append((
                    "table",
                    (f"Evolució anual de la renda mitjana neta per llar al municipi de {selected_mun}", df_rn.T)
                ))
                # Gráfico (barras con etiquetas, sin leyenda si 1 serie)
                items_renda.append((
                    "fig",
                    (f"",
                     mpl_bar(df_rn, ["Renda neta per llar"], title="", ylab="€ per llar",
                             start_year=max(SERIES_START_YEAR, 2015), force_all_xticks=True))
                ))
                
    except Exception:
        pass
    if items_renda:
        sections.append(("Economia — Renda", items_renda))

    # --------- OFERTA DE NOVA CONSTRUCCIÓ (APCE) ---------

    items_oferta = []  # <- important: sempre es reinicia

    if (
        tabla_estudi_oferta is not None
        and isinstance(tabla_estudi_oferta, (list, tuple))
        and len(tabla_estudi_oferta) >= 3
    ):
        try:
            oferta_tables = [
                (
                    f"Habitatges totals a l'estudi d'oferta de nova construcció APCE {LAST_CLOSED_YEAR}-{CURRENT_YEAR_LIMIT} — {selected_mun}",
                    tabla_estudi_oferta[0].set_index("Variable")
                ),
                (
                    f"Habitatges unifamiliars a l'estudi d'oferta de nova construcció APCE {LAST_CLOSED_YEAR}-{CURRENT_YEAR_LIMIT} — {selected_mun}",
                    tabla_estudi_oferta[1].set_index("Variable")
                ),
                (
                    f"Habitatges plurifamiliars a l'estudi d'oferta de nova construcció APCE {LAST_CLOSED_YEAR}-{CURRENT_YEAR_LIMIT} — {selected_mun}",
                    tabla_estudi_oferta[2].set_index("Variable")
                ),
            ]

            for titulo_tab, df_tab in oferta_tables:
                if df_tab is not None and not df_tab.empty:
                    items_oferta.append(("table", (titulo_tab, df_tab)))

            if len(items_oferta) > 0:
                sections.append(("Oferta de nova construcció", items_oferta))

        except Exception:
            pass




    # ==========================
    # 3) Construir y descargar PDF
    # ==========================
    try:
        pdf_bytes = build_location_pdf_ordered(
            location_name=f"{selected_mun}",
            kpis=kpis_pdf,
            sections=sections
        )
        st.download_button(
            label=f"Descarregar informe de mercat — {selected_mun}",
            data=pdf_bytes,
            file_name=f"Informe_{selected_mun}_{datetime.now():%Y%m%d}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"No s'ha pogut generar el PDF per a {selected_mun}: {e}")


#Funciones parte de indicadores idescat


def detect_and_coerce_years(df):
    years = sorted({str(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))}, reverse=True)
    for y in years:
        df[y] = pd.to_numeric(df[y], errors="coerce")
    return years


def add_last_cols(df, years):
    present = [y for y in years if y in df.columns]
    if not present:
        df["last_year"] = None
        df["last_value"] = np.nan
        return df
    arr = df[present].to_numpy(copy=False)
    mask = ~np.isnan(arr)
    idx = mask.argmax(1)
    vals = arr[np.arange(len(df)), idx]
    yrs  = np.array(present)[idx]
    none_mask = ~mask.any(1)
    vals[none_mask] = np.nan
    yrs[none_mask]  = None
    df["last_year"] = yrs
    df["last_value"] = vals
    return df

# Nota: retornen format anglosaxó a propòsit; st_metric els converteix a espanyol.
def fmt_int(x):  return "—" if pd.isna(x) else f"{int(round(float(x))):,}"
def fmt_pct(x):  return "—" if pd.isna(x) else f"{float(x):.2f}"+" %"

def get_year_val(df, vars_, year):
    if not year: return np.nan
    for v in vars_:
        row = df.loc[df["variable"]==v]
        if not row.empty and year in row.columns and pd.notnull(row.iloc[0][year]):
            return float(row.iloc[0][year])
    return np.nan

def latest_year_value(df, vars_, years):
    """(año, valor) más reciente con dato para vars_."""
    for y in years:
        v = get_year_val(df, vars_, y)
        if pd.notnull(v): return y, v
    return None, np.nan

def prev_year_value(df, vars_, cur_year, years):
    """(año, valor) inmediatamente anterior con dato a cur_year para vars_."""
    if not cur_year or cur_year not in years: return None, np.nan
    start = years.index(cur_year) + 1
    for y in years[start:]:
        v = get_year_val(df, vars_, y)
        if pd.notnull(v): return y, v
    return None, np.nan

def sum_age(year, groups, df_pob):
    s=0.0; ok=False
    for cand_es,cand_cat in groups:
        v=get_year_val(df_pob,[cand_es,cand_cat],year)
        if pd.notnull(v): s+=v; ok=True
    return s if ok else np.nan

def latest_year_sum_age(groups, years, df_pob):
    """(año, suma) más reciente con dato para la suma de grupos."""
    for y in years:
        s = sum_age(y, groups, df_pob)
        if pd.notnull(s): return y, s
    return None, np.nan

st.set_page_config(
    page_title="Observatori del Sector APCE",
    page_icon="""data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAA1VBMVEVHcEylpKR6eHaBgH9GREGenJxRT06op6evra2Qj49kYWCbmpqdnJyWlJS+vb1CPzyurKyHhYWMiYl7eXgOCgiPjY10cnJZV1WEgoKCgYB9fXt
    /fHyzsrGUk5OTkZGlo6ONioqko6OLioq7urqysbGdnJuurazCwcHLysp+fHx9fHuDgYGJh4Y4NTJcWVl9e3uqqalcWlgpJyacm5q7urrJyMizsrLS0tKIhoaMioqZmJiTkpKgn5+Bf36WlZWdnJuFg4O4t7e2tbXFxMR3dXTg39/T0dLqKxxpAAAAOHRSTlMA/WCvR6hq/
    v7+OD3U9/1Fpw+SlxynxXWZ8yLp+IDo2ufp9s3oUPII+jyiwdZ1vczEli7waWKEmIInp28AAADMSURBVBiVNczXcsIwEAVQyQZLMrYhQOjV1DRKAomKJRkZ+P9PYpCcfbgze+buAgDA5nf1zL8TcLNamssiPG/
    vt2XbwmA8Rykqton/XVZAbYKTSxzVyvVlPMc4no2KYhFaePvU8fDHmGT93i47Xh8ijPrB/0lTcA3lcGQO7otPmZJfgwhhoytPeKX5LqxOPA9i7oDlwYwJ3p0iYaEqWDdlRB2nkDjgJPA7nX0QaVq3kPGPZq/V6qUqt9BAmVaCUcqEdACzTBFCpcyvFfAAxgMYYVy1sTwAAAAASUVORK5CYII=""",
    layout="wide"
)
def load_css_file(css_file_path):
    with open(css_file_path, encoding="utf-8") as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file(CSS_FILE)

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
apply_theme_css(st.session_state["theme"])

# Àncora del principi de la pàgina (el botó flotant "tornar a dalt" hi apunta).
st.markdown('<div id="dalt"></div>', unsafe_allow_html=True)

# Switch clar/fosc (flotant, part superior dreta). L'àncora buida permet que
# el CSS trobi i posicioni NOMÉS aquest widget (mateix patró que menu-nav-anchor).
def _on_theme_toggle():
    st.session_state["theme"] = "dark" if st.session_state["_theme_switch"] else "light"

st.markdown('<div class="theme-toggle-anchor"></div>', unsafe_allow_html=True)
st.toggle(
    "🌙",
    value=(st.session_state["theme"] == "dark"),
    key="_theme_switch",
    on_change=_on_theme_toggle,
    help="Mode clar / fosc",
)

with open(LOGO_APCE_WEB, "rb") as f:
    data_uri = base64.b64encode(f.read()).decode("utf-8")
with open(LOGO_APCE_WEB_DARK, "rb") as f:
    data_uri_dark = base64.b64encode(f.read()).decode("utf-8")
logo_uri = data_uri_dark if st.session_state["theme"] == "dark" else data_uri
markdown = f"""
<div class="image-apce-container">
<img src="data:image/png;base64,{logo_uri}" alt="APCE Catalunya" class="image-apce">
</div>
"""

# Capçalera: una única targeta (barra de navegació moderna) amb el logo a l'esquerra
# i el menú a la dreta, en lloc del logo gran centrat amb el menú a sota. És el primer
# que veu qualsevol visitant, per això es tracta com una sola peça visual unificada
# (vora + fons + ombra), no dos elements solts un al costat de l'altre.
st.markdown('<div class="header-card-anchor"></div>', unsafe_allow_html=True)
with st.container(border=True):
    logo_col, menu_col = st.columns([1, 3], vertical_alignment="center")
    with logo_col:
        st.markdown(markdown, unsafe_allow_html=True)
    with menu_col:
        # Menú de navegació: un st.radio horitzontal estilitzat com a barra de pestanyes
        # (l'aspecte i el responsive es defineixen a main.css, secció "MENÚ DE NAVEGACIÓ").
        # En ser un element natiu de Streamlit, s'adapta sol a mòbils i pantalles petites,
        # a diferència del component extern option_menu (un iframe que deixava forats en blanc).
        # L'àncora buida permet que el CSS estilitzi NOMÉS aquest radio (i no els altres).
        st.markdown('<div class="menu-nav-anchor"></div>', unsafe_allow_html=True)
        selected_top = st.radio(
            "Menú principal",
            ["Indicadors Territorials", "Estudi d'oferta d'obra nova APCE", "Informe de mercat i sectorial", "Viabilitat financera"],
            horizontal=True,
            label_visibility="collapsed",
        )

# "Indicadors Territorials" agrupa els 7 nivells geogràfics sota un segon menú, perquè
# el menú principal no s'omplís d'entrades a mesura que l'app ha anat creixent. Els blocs
# "if selected == ...:" de cada nivell no es toquen: només canvia com s'omple `selected`.
if selected_top == "Indicadors Territorials":
    st.subheader("INDICADORS TERRITORIALS")
    st.markdown('<div class="indicadors-menu-anchor"></div>', unsafe_allow_html=True)
    selected = st.radio(
        "Indicadors territorials",
        ["Espanya", "Catalunya", "Províncies i àmbits", "Comarques", "Municipis", "Districtes de Barcelona", "Mapa interactiu"],
        horizontal=True,
        label_visibility="collapsed",
        key="indicadors_territorials_menu",
    )
else:
    selected = selected_top

#Trimestre lloguer. Única variable que introduce 0s en lugar de NaNs
max_trim_lloguer = f"{CURRENT_YEAR_LIMIT}-12-31"
date_max_hipo_aux = f"{CURRENT_YEAR_LIMIT}-12-31"
date_max_ciment_aux = f"{CURRENT_YEAR_LIMIT}-12-31"
date_max_euribor = f"{CURRENT_YEAR_LIMIT}-12-31"
date_max_ipc = f"{CURRENT_YEAR_LIMIT}-12-31"
@st.cache_data(show_spinner=False)
def import_data(trim_limit, month_limit):
    with open(DATA_FILE_IDESCAT, 'r') as outfile:
        list_idescat_mun = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
        idescat_muns= list_idescat_mun[0].copy()
    with open(DATA_FILE_CENSO, 'r') as outfile:
        list_censo = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    with open(DATA_FILE_INDICADORS_MUN, 'r', encoding="latin-1") as outfile:
        list_mun_idescat = json.load(outfile)
    df_mun_idescat = pd.DataFrame(list_mun_idescat["municipis"])
    df_pob_ine  = pd.DataFrame(list_mun_idescat.get("poblacio_edat_nacionalitat", []))
    censo_2021= list_censo[0].copy()
    censo_2021['Municipi'] = censo_2021['Municipi'].str.replace("L'", "l'", regex=False)
    rentaneta_mun= list_censo[1].copy()
    rentaneta_mun = _elementwise(rentaneta_mun, lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_mun = rentaneta_mun.apply(_try_num_col)
    rentaneta_mun.columns = rentaneta_mun.columns.str.replace("L'", "l'", regex=False)
    censo_2021_dis= list_censo[2].copy()
    rentaneta_dis = list_censo[3].copy()
    rentaneta_dis = _elementwise(rentaneta_dis, lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_dis = rentaneta_dis.apply(_try_num_col)
    with open(DATA_FILE_SIMPLE, 'r') as outfile:
        list_of_df = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    DT_terr= list_of_df[0].copy()
    DT_mun= list_of_df[1].copy()
    DT_mun_aux= list_of_df[2].copy()
    DT_mun_aux2= list_of_df[3].copy()
    DT_mun_aux3= list_of_df[4].copy()
    DT_dis= list_of_df[5].copy()
    DT_terr_y= list_of_df[6].copy()
    DT_mun_y= list_of_df[7].copy()
    DT_mun_y_aux= list_of_df[8].copy()
    DT_mun_y_aux2= list_of_df[9].copy()
    DT_mun_y_aux3= list_of_df[10].copy()
    DT_dis_y= list_of_df[11].copy()
    DT_monthly= list_of_df[12].copy()
    DT_monthly["Fecha"] = DT_monthly["Fecha"].astype("datetime64[ns]")
    maestro_mun= list_of_df[13].copy()
    maestro_dis= list_of_df[14].copy()

    DT_monthly = DT_monthly[DT_monthly["Fecha"]<=month_limit]
    DT_terr = DT_terr[DT_terr["Fecha"]<=trim_limit]
    DT_mun = DT_mun[DT_mun["Fecha"]<=trim_limit]
    DT_mun_aux = DT_mun_aux[DT_mun_aux["Fecha"]<=trim_limit]
    DT_mun_aux2 = DT_mun_aux2[DT_mun_aux2["Fecha"]<=trim_limit]
    DT_mun_aux3 = DT_mun_aux3[DT_mun_aux3["Fecha"]<=trim_limit]
    DT_mun_pre = pd.merge(DT_mun, DT_mun_aux, how="left", on=["Trimestre","Fecha"])
    DT_mun_pre2 = pd.merge(DT_mun_pre, DT_mun_aux2, how="left", on=["Trimestre","Fecha"])
    DT_mun_def = pd.merge(DT_mun_pre2, DT_mun_aux3, how="left", on=["Trimestre","Fecha"])
    mun_list_aux = list(map(str, maestro_mun.loc[maestro_mun["ADD"] == "SI", "Municipi"].tolist()))
    mun_list = ["Trimestre", "Fecha"] + mun_list_aux
    muns_list = '|'.join(mun_list)
    DT_mun_def = DT_mun_def[[col for col in DT_mun_def.columns if any(mun in col for mun in mun_list)]]
    DT_dis = DT_dis[DT_dis["Fecha"]<=trim_limit]
    DT_mun_y_pre = pd.merge(DT_mun_y, DT_mun_y_aux, how="left", on="Fecha")
    DT_mun_y_pre2 = pd.merge(DT_mun_y_pre, DT_mun_y_aux2, how="left", on="Fecha")
    DT_mun_y_def = pd.merge(DT_mun_y_pre2, DT_mun_y_aux3, how="left", on="Fecha")    
    DT_mun_y_def = DT_mun_y_def[[col for col in DT_mun_y_def.columns if any(mun in col for mun in mun_list)]]

    return([DT_monthly, DT_terr, DT_terr_y, DT_mun_def, DT_mun_y_def, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis, idescat_muns, df_mun_idescat, df_pob_ine, DT_mun_y])
import_data = auto_spinner(import_data)
DT_monthly, DT_terr, DT_terr_y, DT_mun, DT_mun_y, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis, idescat_muns, df_mun_idescat, df_pob_ine, DT_mun_y_all = import_data(f"{CURRENT_YEAR_LIMIT}-05-01", f"{CURRENT_YEAR_LIMIT}-05-01")


# ========== ESTUDI D'OFERTA DE NOVA CONSTRUCCIÓ (font: Atlas) ==========
# Única font: DATA_FILE_ATLAS_OFERTA (BBDD_Atlas_trimmed.json), períodes ATLAS_PERIODES
# ("2025_H1", "2026_H1"). Substitueix l'antic proveïdor (fulls històrics 2019-2025 +
# maestro_estudi), ja no disponible.
@st.cache_data(show_spinner=False)
def _carrega_estudi_oferta_atlas():
    """Llegeix el JSON de l'Atlas (10x més ràpid que l'Excel equivalent) i el redueix a format
    llarg (Any, Municipi, Tipologia, Variable, Valor) — mateix esperit que construir_df_final()
    d'Estudi_oferta_atlas.py, però restringit al nivell de municipi (únic nivell que necessita
    aquesta app)."""
    df = pd.read_json(DATA_FILE_ATLAS_OFERTA)
    df = df[df["period_id"].isin(ATLAS_PERIODES)].copy()
    df["Any"] = pd.to_numeric(df["any"], errors="coerce").astype("Int64")
    df["Municipi"] = df["municipality"].astype("string").str.strip()
    df["TIPOG"] = np.where(
        df["clase_vivienda"].astype("string").str.lower().eq("unifamiliar"),
        "HABITATGES UNIFAMILIARS", "HABITATGES PLURIFAMILIARS"
    )
    df["Preu mitjà"] = pd.to_numeric(df["price"], errors="coerce")
    df["Preu m2 útil"] = pd.to_numeric(df["price_m2_util"], errors="coerce")
    df["Superfície útil"] = pd.to_numeric(df["useful_size"], errors="coerce")
    df = df.dropna(subset=["Municipi", "Any"])

    variables = [
        ("Unitats", lambda g: float(len(g))),
        ("Superfície mitjana (m² útils)", lambda g: g["Superfície útil"].mean()),
        ("Preu mitjà de venda de l'habitatge (€)", lambda g: g["Preu mitjà"].mean()),
        ("Preu de venda per m² útil (€)", lambda g: g["Preu m2 útil"].mean()),
    ]
    tipologies = {
        "TOTAL HABITATGES": None,
        "HABITATGES UNIFAMILIARS": "HABITATGES UNIFAMILIARS",
        "HABITATGES PLURIFAMILIARS": "HABITATGES PLURIFAMILIARS",
    }

    files = []
    for (any_estudi, municipi), grup_mun in df.groupby(["Any", "Municipi"]):
        for tip_label, tip_filtre in tipologies.items():
            grup = grup_mun if tip_filtre is None else grup_mun[grup_mun["TIPOG"] == tip_filtre]
            if grup.empty:
                continue
            for var_label, func in variables:
                files.append({
                    "Any": int(any_estudi), "Municipi": municipi, "Tipologia": tip_label,
                    "Variable": var_label, "Valor": func(grup),
                })
    return pd.DataFrame(files)


@st.cache_data(show_spinner=False)
def table_mun_oferta(Municipi, any_ini, any_fin):
    """Taula per a la UI (es mostra amb .to_html()): files=Any, columnes=(Tipologia, Variable)."""
    df_est = _carrega_estudi_oferta_atlas()
    d = df_est[(df_est["Municipi"] == Municipi) & (df_est["Any"] >= any_ini) & (df_est["Any"] <= any_fin)]
    if d.empty:
        return pd.DataFrame()
    taula = d.pivot(index="Any", columns=["Tipologia", "Variable"], values="Valor")
    taula = taula.sort_index(axis=1, level=[0, 1]).round(0)
    return taula.apply(lambda col: col.map(lambda x: "" if pd.isna(x) else f"{x:,.0f}".replace(",", ".")))


@st.cache_data(show_spinner=False)
def table_mun_oferta_aux(Municipi, anys):
    """Llista de 3 taules (Total, Unifamiliars, Plurifamiliars) per al PDF: files=Variable,
    columnes=un any per columna (comparativa entre anys). Valors numèrics: el format
    espanyol final l'aplica _styled_table_from_df en construir la taula del PDF."""
    df_est = _carrega_estudi_oferta_atlas()
    d = df_est[(df_est["Municipi"] == Municipi) & (df_est["Any"].isin(anys))]
    resultats = []
    for tip in ["TOTAL HABITATGES", "HABITATGES UNIFAMILIARS", "HABITATGES PLURIFAMILIARS"]:
        sub = d[d["Tipologia"] == tip]
        taula = sub.pivot(index="Variable", columns="Any", values="Valor").round(0)
        taula.columns = [str(c) for c in taula.columns]
        resultats.append(taula.reset_index())
    return resultats


def _viab_atlas_preu_oferta(municipi, df_est):
    """Preu de venda per m² útil (€), nombre d'habitatges en oferta i any del darrer període
    de l'Estudi d'Oferta d'obra nova (Atlas) per al municipi donat (tipologia TOTAL HABITATGES).
    Retorna (preu_m2, unitats, any_periode); preu_m2 és None si no hi ha dades del municipi."""
    if df_est.empty:
        return None, 0, None
    any_periode = int(df_est["Any"].max())
    d = df_est[(df_est["Municipi"] == municipi) & (df_est["Any"] == any_periode) & (df_est["Tipologia"] == "TOTAL HABITATGES")]
    if d.empty:
        return None, 0, any_periode
    unitats_serie = d.loc[d["Variable"] == "Unitats", "Valor"]
    preu_serie = d.loc[d["Variable"] == "Preu de venda per m² útil (€)", "Valor"]
    unitats = int(unitats_serie.iloc[0]) if not unitats_serie.empty and pd.notna(unitats_serie.iloc[0]) else 0
    preu_m2 = float(preu_serie.iloc[0]) if not preu_serie.empty and pd.notna(preu_serie.iloc[0]) else None
    return preu_m2, unitats, any_periode


# IMPORTANT — Les funcions tidy_* NO es cachegen: reben els DataFrames grans
# (DT_terr, DT_mun, DT_monthly...) i es criden diverses vegades per rerun. Fer el
# hash d'aquests frames per a la clau de la memòria cau costa ~200 ms per crida,
# mentre que el càlcul (filtrar/renombrar) és ~1 ms. Cachejar-les alentiria molt
# l'app (mesurat: fins a 500× més lent). Es deixen sense decorador expressament.
def tidy_Catalunya_m(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Month"] = output_data['Fecha'].dt.month
    output_data = output_data.dropna()
    output_data = output_data[(output_data["Month"]<=output_data['Month'].iloc[-1])]
    return(output_data.drop(["Data", "Month"], axis=1))

def tidy_Catalunya(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Trimestre"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Trimestre"] + columns_output

    return(output_data.set_index("Trimestre").drop("Data", axis=1))

def tidy_Catalunya_anual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = columns_output
    output_data["Any"] = output_data["Any"].astype(str)
    return(output_data.set_index("Any"))

def tidy_Catalunya_mensual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Fecha"] = output_data["Fecha"].astype(str)
    return(output_data.set_index("Fecha"))

def tidy_present(data_ori, columns_sel, year):
    output_data = data_ori[data_ori[columns_sel]!=0][["Trimestre"] + [columns_sel]].dropna()
    output_data["Trimestre_aux"] = output_data["Trimestre"].str[-1]
    output_data = output_data[(output_data["Trimestre_aux"]<=output_data['Trimestre_aux'].iloc[-1])]
    output_data["Any"] = output_data["Trimestre"].str[0:4]
    output_data = output_data.drop(["Trimestre", "Trimestre_aux"], axis=1)
    output_data = output_data.groupby("Any").mean().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==str(year)]
    output_data = output_data.set_index("Any")
    return(output_data.values[0][0]) if not output_data.empty else np.nan

def tidy_present_monthly(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + [columns_sel]]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha", columns_sel])
    output_data = output_data.set_index("Fecha").groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0]) if not output_data.empty else np.nan

def tidy_present_monthly_aux(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.set_index("Fecha").groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0]) if not output_data.empty else np.nan

def tidy_present_monthly_diff(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.set_index("Fecha").groupby("Any").mean().diff().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0]) if not output_data.empty else np.nan

def tidy_present_level(data_ori, columns_sel, year):
    """Nivell en viu (suma) de `columns_sel` per a `year`, sumant només els
    trimestres/mesos d'aquell any que ja existeixen a `data_ori` — tant si
    l'índex és 'Trimestre' (YYYYTn, taules trimestrals via tidy_Catalunya)
    com si hi ha una columna 'Fecha' (taules mensuals via tidy_Catalunya_m).
    Complementa la taula anual (tidy_Catalunya_anual) quan l'any seleccionat
    encara és parcial i no hi surt."""
    if columns_sel not in data_ori.columns:
        return np.nan
    if "Fecha" in data_ori.columns:
        sub = data_ori[["Fecha", columns_sel]].dropna()
        sub = sub[sub["Fecha"].dt.year == int(year)]
    else:
        idx = data_ori.index.astype(str)
        sub = data_ori.loc[idx.str[:4] == str(year), [columns_sel]].dropna()
    return sub[columns_sel].sum() if not sub.empty else np.nan

# Sense @st.cache_data: es crida ~200 cops per rerun (un cop per mètrica) amb dos
# DataFrames com a arguments; fer-ne el hash costaria ~1 s per rerun mentre que el
# càlcul real són ~30 ms (mesurat: cachejar-la era ~30× més lent).
def indicator_year(df, df_aux, year, variable, tipus, frequency=None):
    # `variable` arriba com a string a uns call sites i com a llista a
    # d'altres (herència del codi original: cada tidy_present_* espera un
    # format diferent). Normalitzem un cop aquí perquè la resta de la
    # funció no hagi de dependre de com l'ha passat qui crida.
    variable_str = variable[0] if isinstance(variable, list) else variable
    variable_list = variable if isinstance(variable, list) else [variable]
    # L'any demanat no surt a la taula anual (df): o bé és l'any en curs,
    # encara no tancat (per a aquest indicador concret), o bé un buit real
    # d'històric. Només tractem com "any obert" el primer cas — quan `year`
    # és l'any en curs o posterior a l'últim any tancat global — per no
    # confondre'l amb un forat genuí enmig de l'històric.
    any_tancat = year in df.index.astype(str).tolist()
    any_obert = (not any_tancat) and (int(year) >= LAST_CLOSED_YEAR)
    # L'any per defecte (LAST_CLOSED_YEAR) sempre fa servir el càlcul en viu
    # a partir de mensual/trimestral (més precís que la mitjana anual
    # precalculada), estigui tancat o no — comportament ja existent abans
    # d'aquesta funció es toqués; any_obert només HI AFEGEIX els anys
    # posteriors encara no tancats.
    usar_calcul_en_viu = any_obert or (year == str(LAST_CLOSED_YEAR))
    if (usar_calcul_en_viu and (frequency=="month") and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present_monthly(df_aux, variable_str, year),2))
    if (usar_calcul_en_viu and (frequency=="month_aux") and (tipus=="var")):
        return(round(tidy_present_monthly_aux(df_aux, variable_list, year),2))
    if (usar_calcul_en_viu and (frequency=="month_aux") and ((tipus=="diff"))):
        return(round(tidy_present_monthly_diff(df_aux, variable_list, year),2))
    if (usar_calcul_en_viu and ((tipus=="var") or (tipus=="diff")) and (df_aux.index.name == "Trimestre")):
        # tidy_present espera una taula trimestral (índex "Trimestre"). Si
        # frequency no s'ha indicat però df_aux és en realitat mensual
        # (índex "Fecha", com passa en algun call site que mai havia arribat
        # a executar aquesta branca fins ara), no ho intentem: es deixa
        # "nan" en comptes de barrejar estructures incompatibles.
        try:
            return(round(tidy_present(df_aux.reset_index(), variable_str, year),2))
        except Exception:
            return np.nan
    if tipus=="level":
        df_level = df[df.index==year][variable_str]
        if not df_level.empty:
            return round(df_level.values[0],2)
        if any_obert:
            # Any en curs sense fila a la taula anual (encara no tancat):
            # acumulat en viu amb els trimestres/mesos ja disponibles.
            valor = tidy_present_level(df_aux, variable_str, year)
            return round(valor,2) if pd.notna(valor) else np.nan
        return np.nan
    if tipus=="var":
        df = df[variable_str].pct_change().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2)) if not df.empty else np.nan
    if tipus=="diff":
        df = df[variable_str].diff().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2)) if not df.empty else np.nan

# ========== GESTIÓ TEMPORAL AUTOMÀTICA ==========
# Cada indicador detecta el seu propi darrer període disponible a partir de
# les dades reals de la seva font (DT_monthly/DT_terr/DT_terr_y i equivalents
# de municipi/districte), en comptes de dependre de datetime.now() o d'un
# únic any/trimestre global. Sense @st.cache_data pel mateix motiu que
# indicator_year: operen sobre columnes ja carregades, el cost real és
# ínfim comparat amb el cost de fer-ne el hash.
def last_valid_period(df, col, date_col="Fecha"):
    """Valor de `date_col` a la darrera fila de `df` on `col` no és NaN
    (Timestamp mensual, string de trimestre, o any). None si no hi ha dada."""
    if col not in df.columns:
        return None
    valid = df.loc[df[col].notna(), date_col]
    return valid.iloc[-1] if not valid.empty else None


def last_closed_year(col, df_annual, df_quarterly=None, df_monthly=None, date_col="Fecha"):
    """Darrer any 'tancat' per a `col` a la taula anual `df_annual`: un any
    compta com a tancat si la seva font de més freqüència (mateixa columna a
    df_quarterly/df_monthly) té dades als 4 trimestres / 12 mesos d'aquest
    any. Si `col` no existeix a cap font de més freqüència (p. ex.
    qualificacions d'HPO), es confia directament en el darrer any no nul de
    df_annual. None si `col` no té cap dada anual."""
    if col not in df_annual.columns:
        return None
    years = sorted(df_annual.loc[df_annual[col].notna(), date_col].astype(int).unique(), reverse=True)
    for year in years:
        if df_quarterly is not None and col in df_quarterly.columns:
            n = df_quarterly.loc[
                (pd.to_datetime(df_quarterly[date_col]).dt.year == year) & df_quarterly[col].notna()
            ].shape[0]
            if n >= 4:
                return year
            continue
        if df_monthly is not None and col in df_monthly.columns:
            n = df_monthly.loc[
                (pd.to_datetime(df_monthly[date_col]).dt.year == year) & df_monthly[col].notna()
            ].shape[0]
            if n >= 12:
                return year
            continue
        return year
    return None


def format_period_label(value, frequency):
    """Etiqueta llegible del darrer període disponible: 'Maig 2026' (mensual),
    '2026 T2' (trimestral) o '2025' (anual)."""
    if value is None:
        return "Sense dades"
    if frequency == "month":
        mesos = ["Gener", "Febrer", "Març", "Abril", "Maig", "Juny", "Juliol",
                 "Agost", "Setembre", "Octubre", "Novembre", "Desembre"]
        ts = pd.Timestamp(value)
        return f"{mesos[ts.month - 1]} {ts.year}"
    if frequency == "quarter":
        return str(value).replace("T", " T")
    return str(int(value))


def annual_upper_bound(col, df_annual=None, df_quarterly=None, df_monthly=None, default=None):
    """Límit superior per a les taules anuals (tidy_Catalunya_anual): l'últim
    any tancat per a `col` (last_closed_year), en comptes del CURRENT_YEAR_LIMIT
    fix. Evita que un any en curs (parcial) aparegui a les taules 'DADES
    ANUALS' com si fos un any complet. Per defecte usa les taules d'Espanya/
    Catalunya (DT_terr_y/DT_terr/DT_monthly); per a municipi/districte,
    passar df_annual=DT_mun_y (o DT_dis_y) i df_quarterly=DT_mun (o DT_dis)."""
    df_annual = df_annual if df_annual is not None else DT_terr_y
    df_quarterly = df_quarterly if df_quarterly is not None else DT_terr
    df_monthly = df_monthly if df_monthly is not None else DT_monthly
    year = last_closed_year(col, df_annual, df_quarterly=df_quarterly, df_monthly=df_monthly)
    return year if year is not None else (default if default is not None else max_year)


def last_available_year(col, df_quarterly=None, df_monthly=None, df_annual=None):
    """Últim any amb ALGUNA dada real (no cal que estigui tancat) per a `col`,
    consultant primer la font trimestral, després la mensual, després
    l'anual (la primera que tingui la columna amb dades). A diferència de
    last_closed_year (que exigeix l'any complet, 4 trimestres/12 mesos),
    aquesta detecta el moment en què comença a haver-hi dada real d'un any
    nou — necessari perquè Espanya/Catalunya solen tenir dada uns mesos
    abans que comarques/municipis/districtes per al mateix indicador."""
    for df, date_col in ((df_quarterly, "Fecha"), (df_monthly, "Fecha"), (df_annual, "Fecha")):
        if df is None or col not in df.columns:
            continue
        valid = df.loc[df[col].notna(), date_col]
        if valid.empty:
            continue
        val = valid.iloc[-1]
        return int(val) if isinstance(val, (int, np.integer)) else int(pd.to_datetime(val).year)
    return None


def year_selector_options(ref_col, df_quarterly=None, df_monthly=None, df_annual=None, start_year=2018):
    """(available_years, index_year) per al desplegable "Selecciona un any"
    d'una pestanya concreta: available_years arriba fins a l'últim any amb
    ALGUNA dada real de `ref_col` (l'indicador propi d'aquella geografia,
    p. ex. iniviv_Barcelona per a Municipis=Barcelona), encara que sigui
    parcial. La selecció per defecte (index_year) es queda a l'últim any
    TANCAT, perquè la pàgina no obri mostrant "nan" sense que l'usuari ho
    triï expressament."""
    last_real_year = last_available_year(ref_col, df_quarterly=df_quarterly, df_monthly=df_monthly, df_annual=df_annual)
    upper = max(last_real_year, LAST_CLOSED_YEAR) if last_real_year is not None else LAST_CLOSED_YEAR
    years = list(range(start_year, upper + 1))
    default = LAST_CLOSED_YEAR if LAST_CLOSED_YEAR in years else years[-1]
    return years, default


@st.cache_data(show_spinner=False)
def concatenate_lists(list1, list2):
    result_list = []
    for i in list1:
        result_element = i+ list2
        result_list.append(result_element)
    return(result_list)


@st.cache_data(show_spinner=False)
def _img_to_data_uri(path):
    """Llegeix una imatge local i la retorna com a base64, per incrustar-la en HTML (p.ex. una
    imatge dins d'un enllaç <a>, cosa que st.image() no permet fer directament)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@st.cache_data(show_spinner=False, max_entries=500)
def _build_download_href(df, filename):
    # Part cara (to_excel + base64, ~46 ms): es cacheja segons el contingut del
    # DataFrame i el nom del fitxer, per no regenerar l'Excel a cada rerun quan
    # les dades no han canviat. Es converteix a numèric perquè Excel ho tracti
    # com a números i s'hi puguin aplicar fórmules directament.
    df = df.copy().apply(_try_num_col)
    towrite = io.BytesIO()
    df.to_excel(towrite, index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    return f"""<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>"""

def filedownload(df, filename):
    # Si rebem un Styler (table_trim / table_year), agafem les dades numèriques
    # crues (.data), que sí que són "hashables" per a la memòria cau.
    if hasattr(df, "data"):
        df = df.data
    return _build_download_href(df, filename)

# ========== PLOTLY HELPERS ==========
def _plotly_layout(title_main, title_y, title_x=None, tickformat=",d", legend=None, **extra):
    """Layout comú per als gràfics go.Figure (title/eixos/llegenda/fons compartits)."""
    yaxis = dict(title=title_y, automargin=True)
    if tickformat:
        yaxis["tickformat"] = tickformat
    layout_kwargs = dict(
        title=dict(text=title_main, font=dict(size=13)),
        yaxis=yaxis,
        legend=legend or dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        separators=",.",  # format espanyol: decimals amb coma, milers amb punt
    )
    if title_x is not None:
        xaxis = dict(title=title_x, automargin=True)
        if title_x == "Trimestre":
            xaxis.update(tickangle=-45, nticks=8)
        layout_kwargs["xaxis"] = xaxis
    layout_kwargs.update(extra)
    return go.Layout(**layout_kwargs)


def _plotly_sparse_ticks(index, max_ticks=10):
    values = list(index)
    if len(values) <= max_ticks:
        return None
    step = max(1, int(np.ceil(len(values) / max_ticks)))
    ticks = values[::step]
    if ticks[-1] != values[-1]:
        ticks.append(values[-1])
    return ticks


@st.cache_data(show_spinner=False)
def line_plotly_pob(df, col, title_main, title_y, title_x="Any"):
    fig = px.line(
        df,
        x="Fecha",
        y=col,
        title=title_main,
        labels={"Fecha": title_x, col: title_y},
        color_discrete_sequence=[GLOBAL_PALETTE["total"]],
        markers=True
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=13)),
        separators=",.",  # format espanyol
    )
    fig.update_yaxes(tickformat=",d")
    return fig


@st.cache_data(show_spinner=False)
def line_plotly(table_n, selection_n, title_main, title_y, title_x="Trimestre", replace_0=False):
    plot_cat = table_n[selection_n]
    if replace_0==True:
        plot_cat = plot_cat.replace(0, np.nan)
    colors = PLOTLY_PALETTE
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Scatter(
            x=plot_cat.index,
            y=plot_cat[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = _plotly_layout(title_main, title_y, title_x=title_x)
    fig = go.Figure(data=traces, layout=layout)
    if title_x == "Trimestre":
        tickvals = _plotly_sparse_ticks(plot_cat.index)
        if tickvals:
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=[str(x) for x in tickvals], tickangle=-45)
    return fig

@st.cache_data(show_spinner=False)
def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=LAST_CLOSED_YEAR):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = PLOTLY_PALETTE[:3]
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = _plotly_layout(title_main, title_y, title_x="Any")
    fig = go.Figure(data=traces, layout=layout)
    return fig
@st.cache_data(show_spinner=False)
def stacked_bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=LAST_CLOSED_YEAR):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = PLOTLY_PALETTE[:3]

    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)

    layout = _plotly_layout(title_main, title_y, title_x="Any", barmode='stack')

    fig = go.Figure(data=traces, layout=layout)
    return fig
@st.cache_data(show_spinner=False)
def area_plotly(table_n, selection_n, title_main, title_y, trim):
    plot_cat = table_n[table_n.index>=trim][selection_n]
    fig = px.area(
        plot_cat,
        x=plot_cat.index,
        y=plot_cat.columns,
        title=title_main,
        color_discrete_sequence=PLOTLY_PALETTE,
    )
    fig.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))
    fig.update_traces(opacity=0.4)
    fig.update_layout(
        _plotly_layout(
            title_main,
            title_y,
            title_x="Trimestre",
            legend=dict(x=0, y=1.18, orientation="h"),
            barmode="stack",
        )
    )
    tickvals = _plotly_sparse_ticks(plot_cat.index)
    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=[str(x) for x in tickvals], tickangle=-45)
    fig.update_layout(legend_title_text="")
    return fig

@st.cache_data(show_spinner=False)
def bar_plotly_demografia(table_n, selection_n, title_main, title_y, year_ini, year_fin=LAST_CLOSED_YEAR):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = PLOTLY_PALETTE_DEMOGRAFIA[:4]
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            text=plot_cat[col],
            textfont=dict(color="white"),
            marker=dict(color=colors[i % len(colors)]),
        )
        traces.append(trace)
    layout = _plotly_layout(title_main, title_y, title_x="Any")
    fig = go.Figure(data=traces, layout=layout)
    return fig

@st.cache_data(show_spinner=False)
def donut_plotly_demografia(table_n, selection_n, title_main, title_y):
    plot_cat = table_n[selection_n]
    plot_cat = plot_cat.set_index("Tamany").sort_index()
    colors = PLOTLY_PALETTE_DEMOGRAFIA
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Pie(
            labels=plot_cat.index,
            values=plot_cat[col],
            name=col,
            hole=0.5,
            marker=dict(colors=colors)
        )
        traces.append(trace)
    layout = _plotly_layout(title_main, title_y, tickformat=None)
    fig = go.Figure(data=traces, layout=layout)
    return fig

@st.cache_data(show_spinner=False)
def table_monthly(data_ori, year_ini, rounded=True):
    data_ori = data_ori.reset_index()
    month_mapping_catalan = {
        1: 'Gener',
        2: 'Febrer',
        3: 'Març',
        4: 'Abril',
        5: 'Maig',
        6: 'Juny',
        7: 'Juliol',
        8: 'Agost',
        9: 'Setembre',
        10: 'Octubre',
        11: 'Novembre',
        12: 'Desembre'
    }

    try:
        output_data = data_ori[data_ori["Data"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Data'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = _elementwise(output_data[numeric_columns], lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "Data"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    except KeyError:
        output_data = data_ori[data_ori["Fecha"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Fecha'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = _elementwise(output_data[numeric_columns], lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "index"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    return(output_data)

def format_dataframes(df, style_n):
    # Format espanyol: milers amb punt i decimals amb coma (style_n=True -> 0 decimals; False -> 1 decimal)
    if style_n==True:
        return(df.style.format(thousands=".", decimal=",", precision=0))
    else:
        return(df.style.format(thousands=".", decimal=",", precision=1))



def table_trim(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    data_ori["Any"] = data_ori["Trimestre"].str.split("T").str[0]
    data_ori["Trimestre"] = data_ori["Trimestre"].str.split("T").str[1]
    data_ori["Trimestre"] = data_ori["Trimestre"] + "T"
    data_ori = data_ori[data_ori["Any"]>=str(year_ini)]
    data_ori = data_ori.replace(0, np.nan)
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = _elementwise(data_ori[numeric_columns], lambda x: round(x, 1))
    output_data = data_ori.set_index(["Any", "Trimestre"]).T#.dropna(axis=1, how="all")
    last_column_contains_all_nans = output_data.iloc[:, -1].isna().all()
    if last_column_contains_all_nans:
        output_data = output_data.iloc[:, :-1]
    else:
        output_data = output_data.copy()
    
    if formated==True:   
        return(format_dataframes(output_data, True))
    else:
        return(format_dataframes(output_data, False))


def table_year(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = _elementwise(data_ori[numeric_columns], lambda x: round(x, 1))
    data_output = data_ori[data_ori["Any"]>=str(year_ini)].T
    data_output.columns = data_output.iloc[0,:]
    data_output = data_output.iloc[1:,:]
    if formated==True:   
        return(format_dataframes(data_output, True))
    else:
        return(format_dataframes(data_output, False))

@st.cache_resource(show_spinner=False)
def load_shp(p):
    s=gpd.read_file(p); 
    s["nom_muni"]=s["nom_muni"].astype(str)
    s["codiine"] = s["codiine"].astype(int)
    s["geometry"]=s.geometry.simplify(8e-4, preserve_topology=True)
    return s
load_shp = auto_spinner(load_shp)
shapefile_mun = load_shp(SHAPEFILE_MUN)

@st.cache_data(show_spinner=False, max_entries=64)
def tmp_map(_DT_mun_y, _shapefile_mun, _maestro_mun, var_prefix, any, fecha_col="Fecha"):
    cols = _DT_mun_y.filter(regex=f"^{var_prefix}").columns
    df_long = (
        _DT_mun_y[[fecha_col] + list(cols)]
        .melt(id_vars=fecha_col, value_vars=cols,
              var_name="variable", value_name="valor")
    )
    df_long["nom_muni"] = df_long["variable"].str.replace(var_prefix, "", regex=False)
    df_long[fecha_col] = pd.to_numeric(df_long[fecha_col], errors="coerce")
    df_long = df_long[df_long[fecha_col] == int(any)][["nom_muni", "valor"]].copy()
    df_long["valor"] = pd.to_numeric(df_long["valor"], errors="coerce")
    df_long = df_long.merge(
        _maestro_mun[["Codi", "Municipi"]],
        left_on="nom_muni",
        right_on="Municipi",
        how="left"
    ).dropna(subset=["Codi"])
    df_long["Codi"] = df_long["Codi"].astype(int)
    df_long["valor"] = df_long["valor"].replace(0, np.nan)
    output = _shapefile_mun.merge(
        df_long[["Codi", "valor"]],
        left_on="codiine",
        right_on="Codi",
        how="left"
    )
    output["valor_fmt"] = output["valor"].map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notnull(x) else "Sense dades")
    return output


def folium_mapa_municipis(map_df, any, name_var):
    dark_mode = st.session_state.get("theme", "light") == "dark"
    m = folium.Map(
        location=[41.75, 1.65],
        zoom_start=8,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("CartoDB positron", name="Clar", control=True, show=not dark_mode).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Fosc", control=True, show=dark_mode).add_to(m)

    folium.Choropleth(
        geo_data=map_df.__geo_interface__,
        data=map_df,
        columns=["codiine", "valor"],
        key_on="feature.properties.codiine",
        fill_color="YlOrRd",
        fill_opacity=0.78,
        line_opacity=0.25,
        line_weight=0.4,
        legend_name=f"{name_var} {any}",
        nan_fill_color="#d9d9d9",
        nan_fill_opacity=0.25,
    ).add_to(m)

    folium.GeoJson(
        map_df.__geo_interface__,
        name="Municipis",
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_muni", "valor_fmt"],
            aliases=["Municipi:", "Valor:"],
            localize=True,
            sticky=False,
        ),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0.35, "color": "#444444"},
        highlight_function=lambda x: {"weight": 2, "color": "#D9773F", "fillOpacity": 0.18},
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# Defining years
max_year= CURRENT_YEAR_LIMIT
# L'any per defecte del selector es detecta a partir de les dades reals
# (Producció nacional, indicador sempre disponible) en comptes de confiar
# cegament en CURRENT_YEAR_LIMIT - 1: si CURRENT_YEAR_LIMIT s'actualitza
# abans que arribin les dades completes del nou any (p. ex. es puja a l'any
# següent però encara no hi ha 12 mesos/4 trimestres tancats), el selector
# segueix oferint per defecte l'últim any que realment té dades completes,
# evitant una selecció per defecte que apunti a una taula buida.
_any_referencia = last_closed_year("iniviv_Nacional", DT_terr_y, df_quarterly=DT_terr, df_monthly=DT_monthly)
if _any_referencia is not None:
    LAST_CLOSED_YEAR = min(LAST_CLOSED_YEAR, _any_referencia)
available_years = list(range(2018, max_year + 1))  # inclou l'any en curs (2018..CURRENT_YEAR_LIMIT) com a opció seleccionable, encara que sigui parcial
index_year = LAST_CLOSED_YEAR  # any seleccionat per defecte al selector: l'últim any tancat (l'any en curs cal triar-lo expressament)

###################################################################### VIABILITAT DE PROMOCIÓ: FUNCIONS ##########################################################################
# Replica la lògica de càlcul de Viabilidad_promocion/APP_Dades.py (codi de
# referència, no es toca). Diferències deliberades respecte l'original:
#  - Sense Google Sheets: tot es calcula en una sola passada dins del mateix
#    rerun (l'estàtic es calcula sense finançament, el dinàmic consumeix
#    aquests totals i retorna els interessos, i llavors es tanca l'estàtic
#    amb el BAI). L'original resolia aquesta circularitat escrivint/llegint
#    de Sheets entre pestanyes.
#  - Les corbes de ponderació trimestral (construcció/vendes/etc.) són
#    totes editables amb un únic st.data_editor, amb valors per defecte
#    raonables, en comptes de venir precarregades d'un Google Sheet extern
#    que aquí no existeix.
#  - Edificabilitat com a inputs simples (superfície construïda, municipi),
#    sense la pestanya de comparació de 3 propostes ni els enllaços a
#    Autodesk Forma.

def _viab_date_to_quarter(date_val):
    d = pd.to_datetime(date_val)
    quarter = (d.month - 1) // 3 + 1
    return f"{d.year}T{quarter}"

def _viab_add_quarters(start_date, num_quarters):
    current_quarter = _viab_date_to_quarter(start_date)
    year, quarter = current_quarter.split("T")
    new_year = int(year) + (int(quarter) + num_quarters - 1) // 4
    new_quarter = (int(quarter) + num_quarters - 1) % 4 + 1
    return f"{new_year}T{new_quarter}"

def _viab_calcula_tir(cashflows):
    """TIR anualitzada (trimestral × 4), igual que calcula_tir() de Viabilidad_promocion."""
    try:
        irr = npf.irr(np.array(cashflows, dtype=float))
        return round(irr * 100 * 4, 2) if pd.notna(irr) else np.nan
    except Exception:
        return np.nan

def _viab_calcula_payback(cashflows_acum):
    """Primer trimestre on el cash flow acumulat és positiu i es manté >=0 en endavant."""
    for i, value in enumerate(cashflows_acum):
        if value > 0 and all(v >= 0 for v in cashflows_acum[i:]):
            return cashflows_acum.index[i]
    return None

def _viab_default_curves(quarters):
    """Corbes de ponderació trimestral per defecte (sumen 1.0 cadascuna),
    editables per l'usuari. quarters: llista de 10 etiquetes ("2026T1"...)."""
    n = len(quarters)
    def _pad(values):
        values = list(values) + [0.0] * (n - len(values))
        return values[:n]
    curves = pd.DataFrame({
        "EVOLUCIÓN DE LA CONSTRUCCIÓN": _pad([0.05, 0.10, 0.20, 0.25, 0.25, 0.15]),
        "EVOLUCIÓN DE LAS VENTAS": _pad([0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05]),
        "SOLAR": _pad([1.0]),
        "ADMINISTRACIÓN PROMOCIÓN": _pad([0.05, 0.10, 0.20, 0.25, 0.25, 0.15]),
        "COMERCIALIZACIÓN DE LA PROMOCIÓN": _pad([0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05]),
        "IVA SOLAR": _pad([1.0]),
        "IVA VENTAS": _pad([0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05]),
        "GASTOS DE CONSTITUCIÓN": _pad([0.0, 0.0, 1.0]),
    }, index=quarters).T
    return curves

def _viab_calcul_estatic(mode, superficie_construida, preciom2, costem2, tipo_interes,
                          rentabilidad_pct=None, preu_solar_manual=None, intereses_hipoteca=0.0):
    """Càlcul estàtic (comptes de resultats) sense financiació encara resolta
    del tot: retorna els totals de gastos/ingressos/BAII i, si ja es coneixen
    els interessos (segona passada, després del dinàmic), també el BAI.
    mode: "rentabilitat" o "preu_solar" (els dos mètodes de l'original)."""
    ingresos = preciom2 * superficie_construida
    edificacion1 = costem2 * superficie_construida
    edificacion2 = VIAB_HONORARIS_PCT * edificacion1
    edificacion3 = VIAB_LLICENCIES_PCT * edificacion1
    edificacion4 = VIAB_GASTOS_LEGALS_PCT * edificacion1
    edificacion5 = VIAB_ALTRES_EDIF_PCT * edificacion1
    admin1 = VIAB_ADMIN_PROMOCIO_PCT * edificacion1
    admin2 = VIAB_COMERCIALITZACIO_PCT * ingresos
    total_edificacion = edificacion1 + edificacion2 + edificacion3 + edificacion4 + edificacion5

    if mode == "rentabilitat":
        solar1 = ((ingresos / (1 + (rentabilidad_pct / 100))) - total_edificacion - admin1 - admin2) / (1 + VIAB_OTROS_SOLAR_PCT)
    else:
        solar1 = preu_solar_manual
    solar2 = VIAB_OTROS_SOLAR_PCT * solar1
    total_solar = solar1 + solar2

    total_gastos = total_solar + total_edificacion + admin1 + admin2
    baii = ingresos - total_gastos

    gastos_constitucio = VIAB_GASTOS_CONSTITUCIO_PCT * VIAB_CREDIT_PCT * ingresos
    total_financiacio = intereses_hipoteca + gastos_constitucio
    bai = baii - total_financiacio

    return {
        "ingresos": ingresos, "solar1": solar1, "solar2": solar2, "total_solar": total_solar,
        "edificacion1": edificacion1, "edificacion2": edificacion2, "edificacion3": edificacion3,
        "edificacion4": edificacion4, "edificacion5": edificacion5, "total_edificacion": total_edificacion,
        "admin1": admin1, "admin2": admin2, "total_gastos": total_gastos, "baii": baii,
        "gastos_constitucio": gastos_constitucio, "total_financiacio": total_financiacio, "bai": bai,
        "recursos_propis": VIAB_RECURSOS_PROPIS_PCT * ingresos, "credit_concedit": VIAB_CREDIT_PCT * ingresos,
    }

def _viab_calcul_dinamic(estatic, curves, quarters, tipo_interes):
    """Taula de cash flows trimestrals (10 columnes T0..T9), replicant
    exactament l'ordre de càlcul de l'original (disposició de crèdit des de
    T2, amortització T6-T9). `curves` és un DataFrame (files=conceptes,
    columnes=quarters) amb pesos que sumen 1.0 per fila."""
    tasa_trim = (tipo_interes / 100) / 4
    df = pd.DataFrame(index=[
        "EVOLUCIÓN DE LAS VENTAS", "IVA VENTAS", "SOLAR", "EDIFICACIÓN",
        "ADMINISTRACIÓN PROMOCIÓN", "COMERCIALIZACIÓN DE LA PROMOCIÓN", "IVA SOLAR",
        "GASTOS DE CONSTITUCIÓN",
    ], columns=quarters, dtype=float)

    df.loc["EVOLUCIÓN DE LAS VENTAS"] = curves.loc["EVOLUCIÓN DE LAS VENTAS"] * estatic["ingresos"]
    df.loc["IVA VENTAS"] = curves.loc["IVA VENTAS"] * (VIAB_IVA_EDIFICACIO_PCT * estatic["edificacion1"])
    df.loc["SOLAR"] = curves.loc["SOLAR"] * estatic["total_solar"]
    df.loc["EDIFICACIÓN"] = curves.loc["EVOLUCIÓN DE LA CONSTRUCCIÓN"] * estatic["total_edificacion"]
    df.loc["ADMINISTRACIÓN PROMOCIÓN"] = curves.loc["ADMINISTRACIÓN PROMOCIÓN"] * estatic["admin1"]
    df.loc["COMERCIALIZACIÓN DE LA PROMOCIÓN"] = curves.loc["COMERCIALIZACIÓN DE LA PROMOCIÓN"] * estatic["admin2"]
    df.loc["IVA SOLAR"] = curves.loc["IVA SOLAR"] * (VIAB_IVA_SOLAR_PCT * estatic["solar1"])
    df.loc["GASTOS DE CONSTITUCIÓN"] = curves.loc["GASTOS DE CONSTITUCIÓN"] * estatic["gastos_constitucio"]

    df.loc["CASH FLOW ANTES FINANCIACIÓN"] = (
        df.loc["EVOLUCIÓN DE LAS VENTAS"] + df.loc["IVA VENTAS"]
        - df.loc["SOLAR"] - df.loc["EDIFICACIÓN"] - df.loc["ADMINISTRACIÓN PROMOCIÓN"]
        - df.loc["COMERCIALIZACIÓN DE LA PROMOCIÓN"] - df.loc["IVA SOLAR"]
    )
    df.loc["CASH FLOW ANTES FINANCIACIÓN ACUM"] = df.loc["CASH FLOW ANTES FINANCIACIÓN"].cumsum()

    for row in ["CREDITO UTILIZADO", "INTERESES SOBRE EL SALDO VIVO", "SALDO VIVO DEL CRÉDITO", "DEVOLUCIONES DEL PRINCIPAL"]:
        df.loc[row] = np.nan

    cols = quarters  # cols[2] = T2, etc. (igual que l'original: la disposició de crèdit comença al 3r trimestre)
    df.loc["CREDITO UTILIZADO", cols[2]] = (
        -df.loc["CASH FLOW ANTES FINANCIACIÓN ACUM", cols[2]] + df.loc["GASTOS DE CONSTITUCIÓN", cols[2]]
        - estatic["recursos_propis"]
    ) / (1 - tasa_trim)
    df.loc["INTERESES SOBRE EL SALDO VIVO", cols[2]] = tasa_trim * df.loc["CREDITO UTILIZADO", cols[2]]
    df.loc["SALDO VIVO DEL CRÉDITO", cols[2]] = df.loc["CREDITO UTILIZADO", cols[2]]

    for i in [3, 4, 5]:
        df.loc["CREDITO UTILIZADO", cols[i]] = (
            -df.loc["CASH FLOW ANTES FINANCIACIÓN", cols[i]] + df.loc["INTERESES SOBRE EL SALDO VIVO", cols[i-1]]
        ) / (1 - tasa_trim)
        df.loc["SALDO VIVO DEL CRÉDITO", cols[i]] = df.loc["CREDITO UTILIZADO", :cols[i]].dropna().sum()
        df.loc["INTERESES SOBRE EL SALDO VIVO", cols[i]] = tasa_trim * df.loc["CREDITO UTILIZADO", cols[i]]

    for i in [6, 7, 8, 9]:
        df.loc["SALDO VIVO DEL CRÉDITO", cols[i]] = 0
        df.loc["INTERESES SOBRE EL SALDO VIVO", cols[i]] = 0

    df.loc["DEVOLUCIONES DEL PRINCIPAL", cols[6]] = VIAB_CREDIT_PCT * df.loc["EVOLUCIÓN DE LAS VENTAS", :cols[6]].dropna().sum()
    for i in [7, 8]:
        df.loc["DEVOLUCIONES DEL PRINCIPAL", cols[i]] = VIAB_CREDIT_PCT * df.loc["EVOLUCIÓN DE LAS VENTAS", cols[i]]
    # cols[9] (DEVOLUCIONES) es queda a NaN de moment, igual que a l'original: es
    # completa amb el fillna(0) de sota (l'última quota no es reparteix explícitament).
    for i in [6, 7, 8, 9]:
        df.loc["CREDITO UTILIZADO", cols[i]] = df.loc["DEVOLUCIONES DEL PRINCIPAL", cols[i]] - df.loc["SALDO VIVO DEL CRÉDITO", cols[i-1]]

    df = df.fillna(0)

    df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN"] = (
        df.loc["CASH FLOW ANTES FINANCIACIÓN"] + df.loc["CREDITO UTILIZADO"]
        - df.loc["GASTOS DE CONSTITUCIÓN"] - df.loc["INTERESES SOBRE EL SALDO VIVO"] - df.loc["DEVOLUCIONES DEL PRINCIPAL"]
    )
    df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN ACUM"] = df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN"].cumsum()

    total_intereses = df.loc["INTERESES SOBRE EL SALDO VIVO"].sum()
    return df, total_intereses

def _viab_fmt_num(value, decimals=0):
    """Formata un número en estil espanyol (punt de milers, coma decimal) per mostrar-lo dins d'un input editable."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")

def _viab_parse_num(text, fallback=0.0):
    """Parseja un text en estil espanyol (1.234,5) a float. Si no és vàlid, retorna el valor anterior."""
    cleaned = re.sub(r"[^\d,.-]", "", str(text)).replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return fallback

def _viab_number_input(label, key, default, min_value=0.0, decimals=0, help=None, placeholder=None):
    """Input numèric amb format espanyol (p.ex. 3.000,00) en lloc del format natiu (3000.00) de
    st.number_input. `default=None` deixa el camp buit (amb `placeholder`) en comptes d'omplir-lo
    amb un valor orientatiu. `key` ha de ser únic per cada context (p.ex. incloure el municipi)
    perquè el camp es reiniciï correctament quan canvia aquest context."""
    val_key = f"{key}__val"
    if val_key not in st.session_state:
        st.session_state[val_key] = float(default) if default is not None else 0.0

    def _on_change():
        parsed = max(min_value, _viab_parse_num(st.session_state[key], st.session_state[val_key]))
        st.session_state[val_key] = parsed
        st.session_state[key] = _viab_fmt_num(parsed, decimals)

    if key not in st.session_state:
        st.session_state[key] = _viab_fmt_num(st.session_state[val_key], decimals) if default is not None else ""

    st.text_input(label, key=key, help=help, on_change=_on_change, placeholder=placeholder)
    return st.session_state[val_key]

###################################################################### SCRIPT PESTAÑAS ##########################################################################
if selected == "Espanya":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Sector residencial","Indicadors econòmics"), horizontal=True)
    with center:
        if selected_type=="Indicadors econòmics":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Índex de Preus al Consum (IPC)", "Consum de ciment","Tipus d'interès", "Hipoteques"], key=101)
        if selected_type=="Sector residencial":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus"], key=201)
    with right:
        # Cada indicador d'aquesta pestanya té la seva pròpia freqüència de
        # publicació (IPC/Euríbor/Hipoteques solen tenir dada abans que
        # Producció/Compravendes/Preus): el selector d'any usa la columna
        # real de l'indicador seleccionat, no sempre "iniviv_Nacional".
        _ref_col_espanya = {
            "Producció": "iniviv_Nacional",
            "Compravendes": "trvivnes",
            "Preus": "prvivlfom_Nacional",
            "Índex de Preus al Consum (IPC)": "IPC_Nacional_x",
            "Consum de ciment": "cons_ciment_Espanya",
            "Tipus d'interès": "Euribor_3m",
            "Hipoteques": "hipon_Nacional",
        }.get(selected_index, "iniviv_Nacional")
        available_years, index_year = year_selector_options(_ref_col_espanya, df_quarterly=DT_terr, df_monthly=DT_monthly, df_annual=DT_terr_y)
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=102)

    if selected_type=="Indicadors econòmics":
        if selected_index=="Índex de Preus al Consum (IPC)":
            st.subheader("ÍNDEX DE PREUS AL CONSUM (IPC)")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2002
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], f"{str(min_year)}-01-01", date_max_ipc,["Data","IPC (Base 2021)","IPC subjacent", "IGC"])

            table_espanya_m["Inflació"] = table_espanya_m["IPC (Base 2021)"].pct_change(12).mul(100)
            table_espanya_m["Inflació subjacent"] = round(table_espanya_m["IPC subjacent"],1)
            table_espanya_m["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_m["IGC"],1)
            table_espanya_m = table_espanya_m.drop(["IPC subjacent", "IGC"], axis=1)
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], min_year, annual_upper_bound("IPC_Nacional_x"),["Any", "IPC (Base 2021)","IPC subjacent", "IGC"])
            table_espanya_y["Inflació"] = table_espanya_y["IPC (Base 2021)"].pct_change(1).mul(100)
            table_espanya_y["Inflació subjacent"] = round(table_espanya_y["IPC subjacent"],1)
            table_espanya_y["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_y["IGC"],1)
            table_espanya_y = table_espanya_y.drop(["IPC subjacent", "IGC"], axis=1)

            if selected_year_n==max_year:
                left, center, right= st.columns((1,1,1))
                with left:
                    st_metric(label="**Inflació** (var. anual)", value=f"""{round(table_espanya_m["Inflació"][-1],1)}%""")
                with center:
                    st_metric(label="**Inflació subjacent** (var. anual)", value=f"""{round(table_espanya_m["Inflació subjacent"][-1],1)}%""")
                with right:
                    st_metric(label="**Índex de Garantia de Competitivitat** (var. anual)", value=f"""{round(table_espanya_m["Índex de Garantia de Competitivitat (IGC)"][-1],1)}%""")
            if selected_year_n!=max_year:
                left, center, right= st.columns((1,1,1))
                with left:
                    st_metric(label="**Inflació** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació"].values[0], 1)}%""")
                with center:
                    st_metric(label="**Inflació subjacent** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació subjacent"].values[0], 1)}%""")
                with right:
                    st_metric(label="**Índex de Garantia de Competitivitat** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Índex de Garantia de Competitivitat (IGC)"].values[0], 1)}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(taula_html_es(table_monthly(table_espanya_m[(table_espanya_m["Data"]>=f"{str(selected_year_n)}-01-01") & (table_espanya_m["Data"]<f"{str(selected_year_n+1)}-01-01")], selected_year_n)), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2023), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            st_plotly_chart(line_plotly(table_espanya_m[table_espanya_m.index>="2015-01-01"], ["Inflació", "Inflació subjacent", "Índex de Garantia de Competitivitat (IGC)"], "Evolució mensual de la inflació (variació anual del IPC) i l'IGC (Índex de Garantia de Competitivitat)", "%",  "Any"), use_container_width=True, responsive=True)
        if selected_index=="Consum de ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_espanya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Espanya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Consum de ciment"])
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Espanya"],  f"{str(min_year)}-01-01", f"{date_max_ciment_aux}",["Data", "Consum de ciment"])
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Espanya"], min_year, annual_upper_bound("cons_ciment_Espanya"),["Any", "Consum de ciment"])
            table_espanya_q = table_espanya_q.dropna(axis=0).div(1000)
            table_espanya_y = table_espanya_y.dropna(axis=0).div(1000)
            st_metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2012), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_espanya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_espanya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (%)", "%", 2012), use_container_width=True, responsive=True)     
        if selected_index=="Tipus d'interès":
            min_year=2008
            st.subheader("TIPUS D'INTERÈS I POLÍTICA MONETÀRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "Euribor_1m", "Euribor_3m",	"Euribor_6m", "Euribor_1y", "tipo_hipo"], f"{str(min_year)}-01-01", date_max_euribor,["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]].reset_index(drop=True).rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"],  f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_q = table_espanya_q[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"], min_year, annual_upper_bound("Euribor_3m"),["Any", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_y = table_espanya_y[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]

            if selected_year_n==max_year:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st_metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 3 mesos"], "diff", "month_aux")} p.b.""")
                with left_center:
                    st_metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 6 mesos"], "diff", "month_aux")} p.b.""")
                with right_center:
                    st_metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 1 any"], "diff", "month_aux")} p.b.""")
                with right:
                    st_metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Tipus d'interès d'hipoteques"], "diff", "month_aux")} p.b.""")
            if selected_year_n!=max_year:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st_metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 3 mesos", "diff", "month")} p.b.""")
                with left_center:
                    st_metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 6 mesos", "diff", "month")} p.b.""")
                with right_center:
                    st_metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 1 any", "diff", "month")} p.b.""")
                with right:
                    st_metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Tipus d'interès d'hipoteques", "diff", "month")} p.b.""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(taula_html_es(table_monthly(table_espanya_m[(table_espanya_m["Fecha"]>=f"{str(selected_year_n)}-01-01") & (table_espanya_m["Fecha"]<f"{str(selected_year_n+1)}-01-01")], selected_year_n)), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2024), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2014, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            selected_columns = ["Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_espanya_m.set_index("Fecha"), selected_columns, "Evolució mensual dels tipus d'interès (%)", "Tipus d'interès (%)",  "Fecha"), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_espanya_y, ["Euríbor a 1 any", "Tipus d'interès d'hipoteques"], "Evolució anual dels tipus d'interès (%)", "Tipus d'interès (%)",  2005), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data", "Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"],  f"{str(min_year)}-01-01", f"{date_max_hipo_aux}",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_q = table_espanya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Nacional", "hipoimp_Nacional"], min_year, annual_upper_bound("hipon_Nacional"),["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_y = table_espanya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            left, right = st.columns((1,1))
            with left:
                st_metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Nombre d'hipoteques", "var", "month_aux")}%""")
            with right:
                st_metric(label="**Import d'hipoteques** (Milers d'euros)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Import d'hipoteques", "var", "month_aux")}%""")

            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2022).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2009, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, rounded=False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_espanya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st_plotly_chart(line_plotly(table_espanya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_espanya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st_plotly_chart(bar_plotly(table_espanya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_type=="Sector residencial":
        if selected_index=="Producció":
            min_year=2008
            st.subheader("PRODUCCIÓ D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats", "Habitatges acabats"])                                                                                                                                                                                                                                                                                                                     
            table_esp = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats", "Habitatges acabats", 
                                                                                                                                                                                                                                                                                            "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                            "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional")+ concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), min_year, annual_upper_bound("iniviv_Nacional"),["Any", "Habitatges iniciats", "Habitatges acabats", "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            left, right = st.columns((1,1))
            with left:
                st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
            with right:
                st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges acabats", "var","month")}%""")

            left, right = st.columns((1,1))    
            with left:
                try:
                    st_metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions provisionals d'HPO**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions definitives d'HPO**", value="No disponible")

            left, right = st.columns((1,1))
            with left:
                try:
                    st_metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value="No disponible")

            with right:
                try:
                    st_metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value="No disponible")
            left, right = st.columns((1,1))
            with left:
                try:
                    st_metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value="No disponible")
            with right:
                try:
                    st_metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
                except IndexError:
                    st_metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value="No disponible")

            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
            selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2014).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_esp, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                st_plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_esp_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2008
            st.subheader("COMPRAVENDES D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_m["Compravendes d'habitatge total"] = table_esp_m["Compravendes d'habitatge de segona mà"] + table_esp_m["Compravendes d'habitatge nou"]
            table_esp = tidy_Catalunya(DT_terr, ["Fecha", "trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp["Compravendes d'habitatge total"] = table_esp["Compravendes d'habitatge de segona mà"] + table_esp["Compravendes d'habitatge nou"]
            table_esp = table_esp[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivses", "trvivnes"], min_year, annual_upper_bound("trvivnes"),["Any", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_y["Compravendes d'habitatge total"] = table_esp_y["Compravendes d'habitatge de segona mà"] + table_esp_y["Compravendes d'habitatge nou"]
            table_esp_y = table_esp_y[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]

            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "var", "month")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var", "month")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "var", "month")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge nou**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2014).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_esp[table_esp.notna()], table_esp.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(stacked_bar_plotly(table_esp_y[table_esp_y.notna()], table_esp.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes", 2008), use_container_width=True, responsive=True)
        if selected_index=="Preus":
                min_year=2008
                st.subheader("VALOR TASAT MITJÀ D'HABITATGE LLIURE €/M\u00b2 (MITMA)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], min_year, annual_upper_bound("prvivlfom_Nacional"),["Any", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                left, right = st.columns((1,1))
                with left:
                    try:
                        st_metric(label=f"""**Preu de l'habitatge lliure** (€/m\u00b2)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure", "level"):,.0f}""")
                    except IndexError:
                        st_metric(label="**Preu de l'habitatge lliure** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st_metric(label=f"""**Preu de l'habitatge lliure nou** (€/m\u00b2)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure nou", "level"),1):,.0f}""")
                    except IndexError:
                        st_metric(label="**Preu de l'habitatge lliure nou** (€/m\u00b2)", value="No disponible")

                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2", 2010), use_container_width=True, responsive=True)
                st.subheader("VARIACIONS ANUALS DE L'ÍNDEX DEL PREU DE L'HABITATGE (INE)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "ipves", "ipvses", "ipvnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "ipves", "ipvses", "ipvnes"], min_year, annual_upper_bound("ipves"),["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label=f"""**Preu d'habitatge total** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu d'habitatge total", "level")} %""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge total** (var. anual)", value="No disponible")
                with center:
                    try:
                        st_metric(label=f"""**Preu d'habitatge de segona mà** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge de segona mà", "level")} %""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge de segona mà** (var. anual)", value="No disponible")
                with right:
                    try:
                        st_metric(label=f"""**Preu d'habitatge nou** (var. anual)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge nou", "level"),1)} %""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge nou** (var. anual)", value="No disponible")

                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Índex trimestral de preus per tipologia d'habitatge (variació anual %)", "%"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Índex anual de preus per tipologia d'habitatge (variació anual %)", "%", 2007), use_container_width=True, responsive=True)

if selected == "Catalunya":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_indicator = st.radio("**Selecciona un tipus d'indicador**", ("Sector residencial","Indicadors econòmics"), horizontal=True, key=301)
        if selected_indicator=="Sector residencial":
            selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True)
    with center:
        if (selected_indicator=="Indicadors econòmics"):
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Costos de construcció", "Mercat laboral", "Consum de Ciment", "Hipoteques"], key=302)
        if ((selected_indicator=="Sector residencial")):
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=303)
        # if (selected_type=="Lloguer") and (selected_indicator=="Sector residencial"):
        #     st.write("")
        
    with right:
        # Cada indicador d'aquesta pestanya té la seva pròpia freqüència de
        # publicació (Hipoteques/Consum de ciment solen tenir dada abans que
        # Producció/Compravendes): el selector d'any usa la columna real de
        # l'indicador seleccionat, no sempre "iniviv_Catalunya".
        _ref_col_catalunya = {
            "Producció": "iniviv_Catalunya",
            "Compravendes": "trvivt_Catalunya",
            "Preus": "prvivt_Catalunya",
            "Superfície": "supert_Catalunya",
            "Costos de construcció": "Costos_edificimitjaneres",
            "Mercat laboral": "emptot_Catalunya",
            "Consum de Ciment": "cons_ciment_Catalunya",
            "Hipoteques": "hipon_Catalunya",
        }.get(selected_index, "iniviv_Catalunya")
        available_years, index_year = year_selector_options(_ref_col_catalunya, df_quarterly=DT_terr, df_monthly=DT_monthly, df_annual=DT_terr_y)
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=305)

    if selected_indicator=="Indicadors econòmics":
        if selected_index=="Mercat laboral":
            st.subheader("MERCAT LABORAL DEL SECTOR DE LA CONSTRUCCIÓ")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["ssunempcons_Catalunya", "aficons_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "emptot_Catalunya", "empcons_Catalunya", "ssunempcons_Catalunya", "aficons_Catalunya"],  f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","emptot_Catalunya", "empcons_Catalunya", "ssunempcons_Catalunya", "aficons_Catalunya"], min_year, annual_upper_bound("emptot_Catalunya"),["Any", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st_metric(label="**Total població ocupada** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "var")}%""")
                st_metric(label="**Atur registrat del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Atur registrat del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Atur registrat del sector de la construcció", "var", "month")}%""")
            with right:
                st_metric(label="**Ocupació del sector de la construcció** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "var")}%""")
                st_metric(label="**Afiliats del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Afiliats del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Afiliats del sector de la construcció", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2012, rounded=True), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2008, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(stacked_bar_plotly(table_catalunya_y, ["Total població ocupada", "Ocupació del sector de la construcció"], "Ocupats totals i del sector de la construcció (milers)", "Milers de persones", 2014), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_catalunya_y, ["Afiliats del sector de la construcció", "Atur registrat del sector de la construcció"], "Afiliats i aturats del sector de la construcció", "Persones", 2014), use_container_width=True, responsive=True)

        if selected_index=="Costos de construcció":
            st.subheader("COSTOS DE CONSTRUCCIÓ PER TIPOLOGIA EDIFICATÒRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2013
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"],  f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"], min_year, annual_upper_bound("Costos_edificimitjaneres"),["Any", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st_metric(label="**Edifici renda normal entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "var")}%""")
                st_metric(label="**Nau industrial** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "var")}%""")
            with right:
                st_metric(label="**Unifamiliar de dos plantes entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "var")}%""")
                st_metric(label="**Edifici d’oficines entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "var")}%""")
            desc_bec_aux = """Els preus per m² construït inclouen l’estudi de seguretat i salut, els honoraris tècnics i permisos d’obra amb un benefici industrial del 20% i despeses generals. Addicionalment, 
            cal comentar que aquests preus fan referència a la província de Barcelona. Si la ubicació de l'obra es troba en una província diferent, la disminució dels preus serà d'un 6% a 8% a Girona, 8% a 10% a Tarragona i del 12% a 15% a Lleida."""
            # desc_bec = f'<div style="text-align: justify">{desc_bec_aux}</div>'
            # st.markdown(desc_bec, unsafe_allow_html=True)
            # st.markdown("")
            # st.markdown(f"""<a href="https://drive.google.com/file/d/1ArRHGTPnDjI2gq9iaGhL4MQK7SbIDNDb/" target="_blank"><button class="download-button">Descarregar BEC</button></a>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2013), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2013, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2013, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_catalunya_q, ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (€/m\u00b2)", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(line_plotly(table_catalunya_q.pct_change(4).mul(100).iloc[4:,:], ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (% var. anual)", "%"), use_container_width=True, responsive=True)

        if selected_index=="Consum de Ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2012
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Consum de ciment"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Catalunya"],  f"{str(min_year)}-01-01", f"{date_max_ciment_aux}",["Data", "Consum de ciment"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Catalunya"], min_year, annual_upper_bound("cons_ciment_Catalunya"),["Any", "Consum de ciment"])

            table_catalunya_q = table_catalunya_q.dropna(axis=0).div(1000)
            table_catalunya_y = table_catalunya_y.dropna(axis=0).div(1000)
            st_metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2018).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, True), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_catalunya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_catalunya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (Milers T.)", "%", 2012), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_catalunya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_m = table_catalunya_m[["Data","Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"],  f"{str(min_year)}-01-01", f"{date_max_hipo_aux}",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_q = table_catalunya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Catalunya", "hipoimp_Catalunya"], min_year, annual_upper_bound("hipon_Catalunya"),["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_y = table_catalunya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            left, right = st.columns((1,1))
            with left:
                st_metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Nombre d'hipoteques", "var", "month_aux")}%""")
            with right:
                st_metric(label="**Import d'hipoteques** (Milers €)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Import d'hipoteques", "var", "month_aux")}%""")
            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2022).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, rounded=False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            left, right = st.columns((1,1))
            with left:
                st_plotly_chart(line_plotly(table_catalunya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st_plotly_chart(line_plotly(table_catalunya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st_plotly_chart(bar_plotly(table_catalunya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st_plotly_chart(bar_plotly(table_catalunya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_indicator=="Sector residencial":
        if selected_type=="Venda":
            if selected_index=="Producció":
                min_year=2008
                st.subheader("PRODUCCIÓ D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_cat_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats", "Habitatges acabats"])    
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                   "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                    "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), min_year, annual_upper_bound("iniviv_Catalunya"),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                              "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                                "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_Catalunya_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                with center:
                    try:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats", "var")}%""")
                with center:
                    try:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions provisionals d'HPO**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions definitives d'HPO**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st_metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value="No disponible")
                # st.markdown("La producció d'habitatge a Catalunya al 2022")
                
                # selected_columns = st.multiselect("**Selecció d'indicadors:**", table_Catalunya.columns.tolist(), default=table_Catalunya.columns.tolist())
                selected_columns_ini = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
                selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2008), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2008), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_Catalunya, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st_plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_Catalunya[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_Catalunya_pluri, table_Catalunya_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_Catalunya_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st_plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_Catalunya[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_Catalunya_uni, table_Catalunya_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            if selected_index=="Compravendes":
                min_year=2014
                st.subheader("COMPRAVENDES D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], min_year, annual_upper_bound("trvivt_Catalunya"),["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_Catalunya,  table_Catalunya.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(stacked_bar_plotly(table_Catalunya_y,  table_Catalunya.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2014), use_container_width=True, responsive=True)
            if selected_index=="Preus":
                min_year=2014
                st.subheader("PREUS PER M\u00b2 CONSTRUÏT")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], min_year, annual_upper_bound("prvivt_Catalunya"),["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")  
                with center:
                    try:
                        st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")  
                with right:
                    try:
                        st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")  
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2014), use_container_width=True, responsive=True)
            if selected_index=="Superfície":
                min_year=2014
                st.subheader("SUPERFÍCIE EN M\u00b2 CONSTRUÏTS")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], min_year, annual_upper_bound("supert_Catalunya"),["Any", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")  
                with center:
                    try:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")  
                with right:
                    try:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")   
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts", 2014), use_container_width=True, responsive=True)   
        if selected_type=="Lloguer":
            st.subheader("MERCAT DE LLOGUER")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2014
            table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivalq_Catalunya", "pmvivalq_Catalunya"], f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivalq_Catalunya",  "pmvivalq_Catalunya"], min_year, annual_upper_bound("trvivalq_Catalunya"),["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_Catalunya, 2021, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_Catalunya, 2014, True), f"{selected_type}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_Catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True), f"{selected_type}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_Catalunya, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer a Catalunya", "€/mes"), use_container_width=True, responsive=True)
                st_plotly_chart(line_plotly(table_Catalunya, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_Catalunya_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer a Catalunya", "€/mes", 2005), use_container_width=True, responsive=True)   
                st_plotly_chart(bar_plotly(table_Catalunya_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer", 2005), use_container_width=True, responsive=True)  
if selected == "Províncies i àmbits":
    prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
    ambit_names = ["Alt Pirineu i Aran","Camp de Tarragona","Comarques centrals","Comarques gironines","Metropolità","Penedès","Ponent","Terres de l'Ebre"]
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True, key=400)
        selected_option = st.radio("**Selecciona un tipus d'àrea geogràfica:**", ["Províncies", "Àmbits territorials"], key=401)
    with center:
        if selected_option=="Províncies":
            selected_geo = st.selectbox('**Selecciona una província:**', prov_names, index= prov_names.index("Barcelona"))
        if selected_option=="Àmbits territorials":
            selected_geo = st.selectbox('**Selecciona un àmbit territorial:**', ambit_names, index= ambit_names.index("Metropolità"), key=402)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=403)
    with right:
        available_years, index_year = year_selector_options(f"iniviv_{selected_geo}", df_quarterly=DT_terr, df_annual=DT_terr_y)
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=404)
    if selected_type=="Venda":
        if selected_option=="Àmbits territorials":
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), min_year, annual_upper_bound(f"iniviv_{selected_geo}"),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st_plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, annual_upper_bound(f"trvivt_{selected_geo}"),["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, annual_upper_bound(f"prvivt_{selected_geo}"),["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""") 
                    except IndexError:
                        st_metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, annual_upper_bound(f"supert_{selected_geo}"),["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
        if selected_option=="Províncies":
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_","finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_geo), min_year, annual_upper_bound(f"iniviv_{selected_geo}"),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats**", value="No disponible")          
                with center:
                    try:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats**", value="No disponible")      
                with center:
                    try:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")

                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st_plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, annual_upper_bound(f"trvivt_{selected_geo}"),["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)     
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, annual_upper_bound(f"prvivt_{selected_geo}"),["Any","Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True)     
                
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, annual_upper_bound(f"supert_{selected_geo}"),["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
                with center:
                    try:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
                with right:
                    try:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st_plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st_plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)

    if selected_type=="Lloguer":
        if selected_option=="Àmbits territorials":
            min_year=2014
            st.subheader(f"MERCAT DE LLOGUER A L'ÀMBIT: {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, annual_upper_bound(f"trvivalq_{selected_geo}"),["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st_plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
        if selected_option=="Províncies":
            min_year=2014
            st.subheader(f"MERCAT DE LLOGUER A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, annual_upper_bound(f"trvivalq_{selected_geo}"),["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st_plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

if selected=="Comarques":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True, key=501)
    with center:
        selected_com = st.selectbox("**Selecciona una comarca:**", sorted(maestro_mun["Comarca"].unique().tolist()), index= sorted(maestro_mun["Comarca"].unique().tolist()).index("Barcelonès"), key=502)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=503)
    with right:
        available_years, index_year = year_selector_options(f"iniviv_{selected_com}", df_quarterly=DT_terr, df_annual=DT_terr_y)
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=504)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats", "Habitatges acabats"])     
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_","finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_com), min_year, annual_upper_bound(f"iniviv_{selected_com}"),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
            table_com_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_com_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats**", value="No disponible")          
            with center:
                try:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_com.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_com.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2008), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2008, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_com[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_com[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_com_pluri, table_com_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_com_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la produció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_com[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_com_uni, table_com_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), min_year, annual_upper_bound(f"trvivt_{selected_com}"),["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
                
            with center:
                try:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge nou**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), min_year, annual_upper_bound(f"prvivt_{selected_com}"),["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")

            with center:
                try:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
            with right:
                try:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021,True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", "Trimestre"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), min_year, annual_upper_bound(f"supert_{selected_com}"),["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
            with center:
                try:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
            with right:
                try:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)

            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A LA COMARCA: {selected_com.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), min_year, annual_upper_bound(f"trvivalq_{selected_com}"),["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")
        with right_col:
            try:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_com}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st_plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
            st_plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st_plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st_plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
if selected=="Municipis":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Venda", "Lloguer", "Altres indicadors"), key=601, horizontal=False)
    with center:
        selected_mun = st.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"), key=602)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=603)
    with right:
        if (selected_type=="Venda") or (selected_type=="Lloguer"):
            available_years, index_year = year_selector_options(f"iniviv_{selected_mun}", df_quarterly=DT_mun, df_annual=DT_mun_y)
            selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=604)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_", "finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_mun), min_year, annual_upper_bound(f"iniviv_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
            table_mun_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_mun_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2008), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2008, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_mun[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_mun[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_mun_pluri, table_mun_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_mun_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_mun[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_mun_uni, table_mun_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construïda", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), min_year, annual_upper_bound(f"trvivt_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible") 
            with right:
                try:
                    st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge nou**", value="No disponible") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_mun = table_mun.replace(0, np.nan)
            table_mun_y = table_mun.reset_index().copy()
            table_mun_y["Any"] = table_mun_y["Trimestre"].str[:4]
            table_mun_y = table_mun_y.drop("Trimestre", axis=1)
            table_mun_y = table_mun_y.groupby("Any").mean()
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value="No disponible") 
            with center:
                try:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value="No disponible") 
            with right:
                try:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value="No disponible") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", "Trimestre", True), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
            try:
                tabla_estudi_oferta = table_mun_oferta(selected_mun, LAST_CLOSED_YEAR, CURRENT_YEAR_LIMIT)
                st.subheader("Estudi d'Oferta de Nova Construcció (APCE). Municipi de " + selected_mun.split(',')[0].strip())
                st.markdown(tabla_estudi_oferta.to_html(), unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
                        <a href="https://estudi-oferta.apcebcn.cat/" 
                        class="button" 
                        target="_blank" 
                        rel="noopener noreferrer">
                        Accedir a l'Estudi d'Oferta
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception:
                pass
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), min_year, annual_upper_bound(f"supert_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
            with center:
                try:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
            with right:
                try:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", "Trimestre", True), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A {selected_mun.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), min_year, annual_upper_bound(f"trvivalq_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")
        with right_col:
            try:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
                st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_mun, 2021, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_mun, 2014, rounded=True), f"{selected_type}_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_mun_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=True), f"{selected_type}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st_plotly_chart(line_plotly(table_mun, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", "Trimestre", True), use_container_width=True, responsive=True)
            st_plotly_chart(line_plotly(table_mun, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st_plotly_chart(bar_plotly(table_mun_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st_plotly_chart(bar_plotly(table_mun_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

    if selected_type=="Altres indicadors":
        st.markdown('<div class="custom-box">DEMOGRAFIA (2021)</div>', unsafe_allow_html=True)
        years_mun = detect_and_coerce_years(df_mun_idescat)
        years_pe  = detect_and_coerce_years(df_pob_ine)
        YEARS = sorted(set(years_mun + years_pe), reverse=True)  # orden descendente global

        df_mun_idescat = add_last_cols(df_mun_idescat, YEARS)
        df_pob_ine  = add_last_cols(df_pob_ine, YEARS)

        nombre_variables = NOMBRE_VARIABLES_IDESCAT
        df_mun_idescat["variable_sin_municipi"] = df_mun_idescat["variable"].str.replace(f"_{selected_mun}$", "", regex=True)
        df_mun_idescat["nombre_largo"] = df_mun_idescat["variable_sin_municipi"].map(nombre_variables)
        sel = (
            df_mun_idescat[df_mun_idescat["variable"].astype(str).str.endswith("_"+selected_mun, na=False)]
            .sort_values("variable")
        )


        POP_KEYS=[f"{selected_mun}_Total_Todas las edades", f"{selected_mun}_Total_Totes les edats"]
        AGE_25_34=[[f"{selected_mun}_Total_De 25 a 29 años", f"{selected_mun}_Total_De 25 a 29 anys"],
                [f"{selected_mun}_Total_De 30 a 34 años", f"{selected_mun}_Total_De 30 a 34 anys"]]
        AGE_35_44=[[f"{selected_mun}_Total_De 35 a 39 años", f"{selected_mun}_Total_De 35 a 39 anys"],
                [f"{selected_mun}_Total_De 40 a 44 años", f"{selected_mun}_Total_De 40 a 44 anys"]]


        # Población total y crecimiento
        pop_year, pop_val = latest_year_value(df_pob_ine, POP_KEYS, YEARS)
        prev_pop_year, pop_prev = prev_year_value(df_pob_ine, POP_KEYS, pop_year, YEARS)
        creix = (pop_val/pop_prev - 1)*100 if pd.notnull(pop_val) and pd.notnull(pop_prev) and pop_prev>0 else np.nan

        # Estructura por edades (cada tramo con su año válido y mismo denominador)
        age2534_year, p2534 = latest_year_sum_age(AGE_25_34, YEARS, df_pob_ine)
        den2534 = get_year_val(df_pob_ine, POP_KEYS, age2534_year)
        pct2534 = (p2534/den2534*100) if pd.notnull(p2534) and pd.notnull(den2534) and den2534>0 else np.nan

        age3544_year, p3544 = latest_year_sum_age(AGE_35_44, YEARS, df_pob_ine)
        den3544 = get_year_val(df_pob_ine, POP_KEYS, age3544_year)
        pct3544 = (p3544/den3544*100) if pd.notnull(p3544) and pd.notnull(den3544) and den3544>0 else np.nan

        # Nacimientos y matrimonios (año propio y mismo denominador)
        naix_year, naix_val = latest_year_value(df_mun_idescat, [f"Naixements_Total_{selected_mun}"], YEARS)
        naix_pop = get_year_val(df_pob_ine, POP_KEYS, naix_year)
        naix_pct = (naix_val/naix_pop*100) if pd.notnull(naix_val) and pd.notnull(naix_pop) and naix_pop>0 else np.nan

        matr_year, matr_val = latest_year_value(df_mun_idescat, [f"Matrimonis_Total_{selected_mun}"], YEARS)
        matr_pop = get_year_val(df_pob_ine, POP_KEYS, matr_year)
        matr_pct = (matr_val/matr_pop*100) if pd.notnull(matr_val) and pd.notnull(matr_pop) and matr_pop>0 else np.nan

        subset_tamaño_mun = censo_2021[censo_2021["Municipi"] == selected_mun][["1", "2", "3", "4", "5 o más"]]
        subset_tamaño_mun_aux = subset_tamaño_mun.T.reset_index()
        subset_tamaño_mun_aux.columns = ["Tamany", "Llars"]
        left, right = st.columns((1,1))
        with left:
            st_metric("Grandària de la llar més freqüent", value=censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño_hogar_frecuente"].values[0])
            st_metric("Proporció de població nacional", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],2):,.0f}%""")
            st_metric(label=f"Població total ({pop_year})", value=fmt_int(pop_val))
            st_metric(label=f"Població 25–34 anys (% sobre total) ({age2534_year})", value=fmt_pct(pct2534))
            st_metric(
                label=f"Nombre de naixements ({int(sel[sel['nombre_largo']=='Nombre de naixements']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de naixements']['last_value'].values[0])
            )
            st_metric(
                label=f"Nombre de matrimonis ({int(sel[sel['nombre_largo']=='Nombre de matrimonis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de matrimonis']['last_value'].values[0])
            )

        with right:
            st_metric("Grandària mitjà de la llar", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño medio del hogar"].values[0],2)}""")
            st_metric("Proporció de població estrangera", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],1)}%""")
            st_metric("Proporció de població amb educació superior", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Porc_Edu_superior"].values[0],1)}%""")
            st_metric(label=f"Població 35–44 anys (% sobre total) ({age3544_year})", value=fmt_pct(pct3544))
            st_metric(label=f"Naixements sobre població ({naix_year})", value=fmt_pct(naix_pct))
            st_metric(label=f"Matrimonis sobre població ({matr_year})", value=fmt_pct(matr_pct))
        if f"poptottine_{selected_mun}" in DT_mun_y.columns and not DT_mun_y[f"poptottine_{selected_mun}"].isna().all():
            st_plotly_chart(
                line_plotly_pob(
                    DT_mun_y[["Fecha", f"poptottine_{selected_mun}"]],
                    f"poptottine_{selected_mun}",
                    f"Evolució anual de la població a {selected_mun}",
                    "Habitants"
                ),
                use_container_width=True
            )
        st.markdown("<div class='custom-box'>ECONOMIA, RENDA I ALTRES</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st_metric("Renda neta per llar (2021)", value=f"""{(rentaneta_mun["rentanetahogar_" + selected_mun].values[-1]):,.0f}""")
            st_metric(
                label=f"Nombre de pensionistes ({int(sel[sel['nombre_largo']=='Nombre de pensionistes']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de pensionistes']['last_value'].values[0])
            )
            st_metric(
                label=f"Parc total de vehicles ({int(sel[sel['nombre_largo']=='Parc total de vehicles']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Parc total de vehicles']['last_value'].values[0])
            )
            st_plotly_chart(bar_plotly_demografia(rentaneta_mun.rename(columns={"Año":"Any"}).set_index("Any"), ["rentanetahogar_" + selected_mun], "Evolució anual de la renda mitjana neta", "€", 2015), use_container_width=True, responsive=True)
        with right:
            st_metric(
                label=f"Base imposable mitjana de l’IRPF (€) ({int(sel[sel['nombre_largo']=='Base imposable mitjana de l’IRPF (€)']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Base imposable mitjana de l’IRPF (€)']['last_value'].values[0])
            )
            _ibi_col = "IBI_quota_" + selected_mun
            _ibi_data = idescat_muns[["Any", _ibi_col]].dropna() if _ibi_col in idescat_muns.columns else idescat_muns.iloc[0:0]
            if not _ibi_data.empty:
                st_metric(f"Quota integra del Impost sobre Bèns Immobles (IBI) ({_ibi_data['Any'].values[0]})", value=f"""{int(_ibi_data[_ibi_col].values[0]):,.0f}""")
            else:
                st_metric("Quota integra del Impost sobre Bèns Immobles (IBI)", value="No disponible")
            st_metric(
                label=f"Residus municipals per càpita (kg/hab/dia) ({int(sel[sel['nombre_largo']=='Residus municipals per càpita (kg/hab/dia)']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Residus municipals per càpita (kg/hab/dia)']['last_value'].values[0])
            )
            st_plotly_chart(donut_plotly_demografia(subset_tamaño_mun_aux,["Tamany", "Llars"], "Distribució del nombre de membres per llar", "Llars"), use_container_width=True, responsive=True)
        st.markdown("<div class='custom-box'>CARACTERÍSTIQUES DEL PARC D'HABITATGE (2021)</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st_metric("Proporció d'habitatges en propietat", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_propiedad"].values[0],1)}%""")
            st_metric("Proporció d'habitatges principals", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
            st_metric("Edat mitjana dels habitatges", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Edad media"].values[0],1)}""")

        with right:
            st_metric("Proporció d'habitatges en lloguer", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_alquiler"].values[0], 1)}%""")
            st_metric("Proporció d'habitatges no principals", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
            st_metric("Superfície mitjana dels habitatges", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Superficie media"].values[0],1)}""")
        st.markdown("<div class='custom-box'>MERCAT LABORAL</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st_metric(
                label=f"Afiliats a la Seguretat Social – Agricultura ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Agricultura']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Agricultura']['last_value'].values[0])
            )
            st_metric(
                label=f"Afiliats a la Seguretat Social – Construcció ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Construcció']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Construcció']['last_value'].values[0])
            )
            st_metric(
                label=f"Atur registrat – Agricultura ({int(sel[sel['nombre_largo']=='Atur registrat – Agricultura']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Agricultura']['last_value'].values[0])
            )
            st_metric(
                label=f"Atur registrat – Construcció ({int(sel[sel['nombre_largo']=='Atur registrat – Construcció']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Construcció']['last_value'].values[0])
            )

            st_metric(
                label=f"Atur registrat – Indústria ({int(sel[sel['nombre_largo']=='Atur registrat – Indústria']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Indústria']['last_value'].values[0])
            )
            st_metric(
                label=f"Població activa ({int(sel[sel['nombre_largo']=='Població activa']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població activa']['last_value'].values[0])
            )
            st_metric(
                label=f"Població inactiva ({int(sel[sel['nombre_largo']=='Població inactiva']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població inactiva']['last_value'].values[0])
            )
        with right:
            st_metric(
                label=f"Afiliats a la Seguretat Social – Indústria ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Indústria']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Indústria']['last_value'].values[0])
            )
            st_metric(
                label=f"Afiliats a la Seguretat Social – Serveis ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Serveis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Serveis']['last_value'].values[0])
            )
            st_metric(
                label=f"Afiliats a la Seguretat Social – Total ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Total']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Total']['last_value'].values[0])
            )

            st_metric(
                label=f"Atur registrat – Serveis ({int(sel[sel['nombre_largo']=='Atur registrat – Serveis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Serveis']['last_value'].values[0])
            )
            st_metric(
                label=f"Atur registrat – Total ({int(sel[sel['nombre_largo']=='Atur registrat – Total']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Total']['last_value'].values[0])
            )
            st_metric(
                label=f"Població ocupada ({int(sel[sel['nombre_largo']=='Població ocupada']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població ocupada']['last_value'].values[0])
            )
            st_metric(
                label=f"Població desocupada ({int(sel[sel['nombre_largo']=='Població desocupada']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població desocupada']['last_value'].values[0])
            )
if selected=="Districtes de Barcelona":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Venda", "Lloguer", "Demografia i parc d'habitatge"), key=701, horizontal=False)
    with center:
        selected_dis = st.selectbox("**Selecciona un districte de Barcelona:**", maestro_dis["Districte"].unique(), key=702)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=703)
    with right:
        if (selected_type=="Venda") or (selected_type=="Lloguer"):
            available_years, index_year = year_selector_options(f"iniviv_{selected_dis}", df_quarterly=DT_dis, df_annual=DT_dis_y)
            selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=704)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2011
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), min_year, annual_upper_bound(f"iniviv_{selected_dis}", df_annual=DT_dis_y, df_quarterly=DT_dis),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            # table_dis_pluri = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            # table_dis_uni = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats**", value="0")
            with center:
                try:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            with left:
                try:
                    st_metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats**", value="0")
            with center:
                try:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")           
            with right:
                try:
                    st_metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st_metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2014), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_dis[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_dis[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_dis_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st_plotly_chart(area_plotly(table_dis[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), min_year, annual_upper_bound(f"trvivt_{selected_dis}", df_annual=DT_dis_y, df_quarterly=DT_dis),["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st_metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with right:
                try:
                    st_metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                except IndexError:
                    st_metric(label="**Compravendes d'habitatge total**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_dis_y, table_dis_y.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2017), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), min_year, annual_upper_bound(f"prvivt_{selected_dis}", df_annual=DT_dis_y, df_quarterly=DT_dis),["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
            with center:
                try:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
            with right:
                try:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st_metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")  
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", "Trimestre",True), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", 2017), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), min_year, annual_upper_bound(f"supert_{selected_dis}", df_annual=DT_dis_y, df_quarterly=DT_dis),["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")  
            with center:
                try:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")  
            with right:
                try:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st_metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")  
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st_plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", "Trimestre", True), use_container_width=True, responsive=True)
            with right_col:
                st_plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", 2017), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        st.subheader(f"MERCAT DE LLOGUER A {selected_dis.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        min_year=2014
        table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), min_year, annual_upper_bound(f"trvivalq_{selected_dis}", df_annual=DT_dis_y, df_quarterly=DT_dis),["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st_metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Nombre de contractes de lloguer**", value="No disponible")   
        with right_col:
            try:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st_metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")   
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_dis, 2021, True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_dis, 2014, True), f"{selected_type}_{selected_dis}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_dis_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=True), f"{selected_type}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st_plotly_chart(line_plotly(table_dis, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", "Trimestre", True), use_container_width=True, responsive=True)
            st_plotly_chart(line_plotly(table_dis, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st_plotly_chart(bar_plotly(table_dis_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st_plotly_chart(bar_plotly(table_dis_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

    if selected_type=="Demografia i parc d'habitatge":
        st.markdown(f'<div class="custom-box">DEMOGRAFIA Y RENDA (2021)</div>', unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            subset_tamaño_dis = censo_2021_dis[censo_2021_dis["Distrito"] == selected_dis][["1", "2", "3", "4", "5 o más"]]
            subset_tamaño_dis_aux = subset_tamaño_dis.T.reset_index()
            subset_tamaño_dis_aux.columns = ["Tamany", "Llars"]
            max_column = subset_tamaño_dis.idxmax(axis=1).values[0]
            st_metric("Grandària de la llar més freqüent", value=max_column)
            st_metric("Proporció de població nacional", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0]*100,0)}%""")
            st_metric("Renda neta per llar", value=f"""{(rentaneta_dis["rentahogar_" + selected_dis].values[-1]):,.0f}""")
        with right:
            st_metric("Grandària mitjà de la llar", value=f"""{censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Tamaño medio del hogar"].values[0]}""")
            st_metric("Proporció de població estrangera", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0],2)*100}%""")
            st_metric("Proporció de població amb educació superior", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_edusuperior"].values[0]*100,1)}%""")

        st.markdown(f"<div class='custom-box'>CARACTERÍSTIQUES DEL PARC D'HABITATGE (2021)</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st_metric("Proporció d'habitatges en propietat", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_propiedad"].values[0],1)}%""")
            st_metric("Proporció d'habitatges principals", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
            st_metric("Edat mitjana dels habitatges", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Edad media"].values[0],1)}""")
            st_plotly_chart(bar_plotly_demografia(rentaneta_dis.rename(columns={"Año":"Any"}).set_index("Any"), ["rentahogar_" + selected_dis], "Evolución anual de la renta media neta anual", "€", 2015), use_container_width=True, responsive=True)
        with right:
            st_metric("Proporció d'habitatges en lloguer", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_alquiler"].values[0], 1)}%""")
            st_metric("Proporció d'habitatges no principals", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
            st_metric("Superfície mitjana dels habitatges", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Superficie Media"].values[0],1)}""")
            st_plotly_chart(donut_plotly_demografia(subset_tamaño_dis_aux,["Tamany", "Llars"], "Distribució del nombre de membres per llar", "Llars"), use_container_width=True, responsive=True)

if selected=="Mapa interactiu":
    st.subheader("MAPA INTERACTIU D'INDICADORS MUNICIPALS")
    opcions = {
        "Habitatges iniciats": "iniviv_",
        "Habitatges acabats": "finviv_",
        "Compravendes d'obra nova": "trvivn_",
        "Compravendes de segona mà": "trvivs_",
        "Compravendes totals": "trvivt_",
        "Preu obra nova per m² construït": "prvivn_",
        "Preu segona mà per m² construït": "prvivs_",
        "Preu total per m² construït": "prvivt_",
        "Superfície mitjana total": "supert_",
        "Renda mitjana de lloguer": "pmvivalq_",
    }
    left, right = st.columns(2)
    with left:
        label = st.selectbox("**Selecciona un indicador:**", list(opcions.keys()), key="map_indicador")
    with right:
        anys_mapa, index_year_mapa = year_selector_options("iniviv_Catalunya", df_quarterly=DT_terr, df_monthly=DT_monthly, df_annual=DT_terr_y, start_year=2015)
        any_mapa = st.selectbox("**Selecciona un any:**", anys_mapa, index=anys_mapa.index(index_year_mapa), key="map_any")

    var_prefix = opcions[label]
    map_df = tmp_map(DT_mun_y_all, shapefile_mun, maestro_mun, var_prefix, any_mapa)
    st_folium(
        folium_mapa_municipis(map_df, any_mapa, label),
        use_container_width=True,
        height=720,
        returned_objects=[],
        key=f"mapa_municipis_{var_prefix}_{any_mapa}",
    )

if selected == "Informe de mercat i sectorial":
    st.subheader("INFORME DE MERCAT PER MUNICIPI")
    left, right = st.columns((1, 1))
    with left:
        selected_mun = st.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"), key=602)
    with right:
        st.write("**Descarrega el informe complet del municipi seleccionat:**")
        if st.button("📄 Descarregar informe PDF"):
            with st.spinner(f"Generant informe per a {selected_mun}..."):
                min_year=2008
                #Producció
                table_mun_prod = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_mun_prod_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_", "finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_mun), min_year, annual_upper_bound(f"iniviv_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
                table_mun_prod_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_mun_prod_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                selected_columns_ini = [col for col in table_mun_prod.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_mun_prod.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                # --- Compravendes ---
                table_mun_tr = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",
                    ["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]
                )
                table_mun_tr_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun),
                    min_year, annual_upper_bound(f"trvivt_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),
                    ["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]
                )

                # --- Preus (no sobreescribir table_mun) ---
                table_mun_pr = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",
                    ["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"]
                ).replace(0, np.nan)
                table_mun_pr_y = table_mun_pr.reset_index().copy()
                table_mun_pr_y["Any"] = table_mun_pr_y["Trimestre"].str[:4]
                table_mun_pr_y = table_mun_pr_y.drop("Trimestre", axis=1).groupby("Any").mean()

                # --- Superfície ---
                table_mun_sup = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-12-31",
                    ["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"]
                )
                table_mun_sup_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun),
                    min_year, annual_upper_bound(f"supert_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),
                    ["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"]
                )

                # --- Lloguer ---
                table_mun_llog = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun),
                    f"{str(min_year)}-01-01", max_trim_lloguer,
                    ["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"]
                )
                table_mun_llog_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun),
                    min_year, annual_upper_bound(f"trvivalq_{selected_mun}", df_annual=DT_mun_y, df_quarterly=DT_mun),
                    ["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"]
                )

                years_mun = detect_and_coerce_years(df_mun_idescat)
                years_pe  = detect_and_coerce_years(df_pob_ine)
                YEARS = sorted(set(years_mun + years_pe), reverse=True)  # orden descendente global

                df_mun_idescat = add_last_cols(df_mun_idescat, YEARS)
                df_pob_ine  = add_last_cols(df_pob_ine, YEARS)

                try:
                    tabla_estudi_oferta = table_mun_oferta_aux(selected_mun, [LAST_CLOSED_YEAR, CURRENT_YEAR_LIMIT])
                except:
                    tabla_estudi_oferta = None
                nombre_variables = NOMBRE_VARIABLES_IDESCAT
                df_mun_idescat["variable_sin_municipi"] = df_mun_idescat["variable"].str.replace(f"_{selected_mun}$", "", regex=True)
                df_mun_idescat["nombre_largo"] = df_mun_idescat["variable_sin_municipi"].map(nombre_variables)
                sel = (
                    df_mun_idescat[df_mun_idescat["variable"].astype(str).str.endswith("_"+selected_mun, na=False)]
                    .sort_values("variable")
                )


                generar_pdf_municipi_tot(
                    selected_mun=selected_mun,
                    # Producció
                    table_mun_prod=table_mun_prod,
                    table_mun_prod_y=table_mun_prod_y,
                    table_mun_prod_pluri=table_mun_prod_pluri,
                    table_mun_prod_uni=table_mun_prod_uni,
                    selected_columns_ini=selected_columns_ini,
                    selected_columns_fin=selected_columns_fin,
                    # Compravendes
                    table_mun_tr=table_mun_tr,
                    table_mun_tr_y=table_mun_tr_y,
                    # Preus
                    table_mun_pr=table_mun_pr,
                    table_mun_pr_y=table_mun_pr_y,
                    # Superfície
                    table_mun_sup=table_mun_sup,
                    table_mun_sup_y=table_mun_sup_y,
                    # Lloguer
                    table_mun_llog=table_mun_llog,
                    table_mun_llog_y=table_mun_llog_y,
                    # Altres indicadors (ya los cargas en tu app)
                    censo_2021=censo_2021,
                    DT_mun_y=DT_mun_y,
                    idescat_muns=idescat_muns,
                    rentaneta_mun=rentaneta_mun,
                    tabla_estudi_oferta = tabla_estudi_oferta
                )

    st.markdown("")
    st.subheader("INFORMES SECTORIALS APCE")
    cols_informes = st.columns(len(INFORMES_SECTORIALS))
    for col_informe, informe in zip(cols_informes, INFORMES_SECTORIALS):
        with col_informe:
            st.markdown(
                f'<a href="{informe["url"]}" target="_blank" rel="noopener noreferrer" class="informe-sectorial-link">'
                f'<img src="data:image/jpeg;base64,{_img_to_data_uri(informe["img"])}" alt="Informe Sectorial {informe["any"]}" class="informe-sectorial-img">'
                f'<span class="informe-sectorial-caption">Informe Sectorial {informe["any"]}</span>'
                f'</a>',
                unsafe_allow_html=True,
            )

if selected == "Viabilitat financera":
    st.subheader("VIABILITAT FINANCERA")
    st.markdown(
        '<div class="viab-toc">'
        '<a href="#viab-inputs">Dades d\'entrada</a>'
        '<a href="#viab-estatic">Anàlisi estàtic</a>'
        '<a href="#viab-dinamic">Anàlisi dinàmic</a>'
        '<a href="#viab-resum">Resum de resultats</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div id="viab-inputs" class="viab-anchor"></div>', unsafe_allow_html=True)

    left, center, right = st.columns((1, 1, 1))
    with left:
        viab_mun = st.selectbox(
            "**Municipi del solar:**",
            maestro_mun[maestro_mun["ADD"] == "SI"]["Municipi"].unique(),
            index=maestro_mun[maestro_mun["ADD"] == "SI"]["Municipi"].tolist().index("Barcelona"),
            key="viab_mun",
        )
    with center:
        viab_superficie = _viab_number_input("**Superfície construïda (m²):**", "viab_superficie", default=3000.0, min_value=0.0, decimals=0)
    with right:
        viab_data_inici = st.date_input("**Data d'inici de l'operació:**", value=datetime.now(), key="viab_data_inici")

    # Dades de mercat ja carregades a l'app (Euríbor, BEC, preu m² per municipi):
    # s'usen com a valor per defecte, editable, en comptes que l'usuari les busqui a mà.
    _euribor_ma12 = DT_monthly[["Fecha", "Euribor_1y"]].dropna().set_index("Fecha")["Euribor_1y"].rolling(window=12).mean().dropna()
    _viab_tipo_interes_default = round(float(_euribor_ma12.iloc[-1]) + 1, 2) if not _euribor_ma12.empty else 3.0

    _bec_ma4 = DT_terr[["Fecha", "Costos_edificimitjaneres"]].dropna().set_index("Fecha")["Costos_edificimitjaneres"].rolling(window=4).mean().dropna()
    _viab_costem2_default = round(float(_bec_ma4.iloc[-1]), 1) if not _bec_ma4.empty else 1000.0

    # Preu de venda per m²: font única l'Estudi d'Oferta d'obra nova (Atlas), mateixa
    # font que la resta de l'app (veure _carrega_estudi_oferta_atlas). Es descarta el
    # valor orientatiu si l'oferta d'obra nova al municipi és massa reduïda per ser fiable.
    _viab_df_est = _carrega_estudi_oferta_atlas()
    _viab_atlas_preu, _viab_atlas_unitats, _viab_atlas_any = _viab_atlas_preu_oferta(viab_mun, _viab_df_est)
    _viab_preu_fiable = _viab_atlas_preu is not None and _viab_atlas_unitats >= VIAB_MIN_UNITATS_OFERTA
    _viab_preciom2_default = int(round(_viab_atlas_preu, 0)) if _viab_preu_fiable else None

    left, center, right = st.columns((1, 1, 1))
    with left:
        viab_tipo_interes = _viab_number_input("**Tipus d'interès (%)** — Euríbor 1 any (mitjana 12m) + 1%", "viab_tipo_interes", default=_viab_tipo_interes_default, min_value=0.0, decimals=2)
    with center:
        viab_costem2 = _viab_number_input("**Cost mitjà del m² construït (BEC)**", "viab_costem2", default=_viab_costem2_default, min_value=0.0, decimals=2)
    with right:
        viab_preciom2 = _viab_number_input(
            f"**Preu de venda per m² a {viab_mun}**", f"viab_preciom2_{viab_mun}", default=_viab_preciom2_default,
            min_value=0.0, decimals=0, placeholder=None if _viab_preu_fiable else "Introdueix el preu manualment",
        )
        if _viab_preu_fiable:
            st.caption(f"{_viab_atlas_unitats} habitatges nous en oferta (font: Estudi d'Oferta d'obra nova APCE, informe {_viab_atlas_any} H1).")
        else:
            st.caption(f"Avís: només {_viab_atlas_unitats} habitatges nous en oferta al municipi (mínim {VIAB_MIN_UNITATS_OFERTA} per a un preu orientatiu fiable). Introdueix el preu manualment.")

    viab_metode = st.radio("**Mètode de càlcul del sòl:**", ("Fixar rendibilitat abans d'impostos i interessos", "Fixar preu del sòl"), horizontal=True, key="viab_metode")
    if viab_metode == "Fixar rendibilitat abans d'impostos i interessos":
        viab_rendibilitat = st.slider("**Rendibilitat objectiu (%)**", 0, 50, value=10, key="viab_rendibilitat")
        viab_preu_solar_manual = None
    else:
        viab_rendibilitat = None
        viab_preu_solar_manual = _viab_number_input("**Cost del sòl (€)**", "viab_preu_solar_manual", default=1000000.0, min_value=0.0, decimals=0)

    quarters = [_viab_add_quarters(viab_data_inici, i) for i in range(VIAB_MAX_TRIM)]
    st.markdown(f'<div class="custom-box">Trimestre d\'inici: {quarters[0]}</div>', unsafe_allow_html=True)

    with st.expander("Corbes d'evolució trimestral (% per trimestre, editable)"):
        _viab_curves_pct = (_viab_default_curves(quarters) * 100).round(1)
        _viab_curves_edited = st.data_editor(_viab_curves_pct, key="viab_curves_editor")
        curves = _viab_curves_edited / 100
        row_sums = curves.sum(axis=1).replace(0, 1)
        curves = curves.div(row_sums, axis=0)  # normalitza per si l'usuari desquadra una fila

    _viab_mode = "rentabilitat" if viab_metode == "Fixar rendibilitat abans d'impostos i interessos" else "preu_solar"
    estatic_pre = _viab_calcul_estatic(
        _viab_mode, viab_superficie, viab_preciom2, viab_costem2, viab_tipo_interes,
        rentabilidad_pct=viab_rendibilitat, preu_solar_manual=viab_preu_solar_manual, intereses_hipoteca=0.0,
    )
    dinamic_df, total_intereses = _viab_calcul_dinamic(estatic_pre, curves, quarters, viab_tipo_interes)
    estatic = _viab_calcul_estatic(
        _viab_mode, viab_superficie, viab_preciom2, viab_costem2, viab_tipo_interes,
        rentabilidad_pct=viab_rendibilitat, preu_solar_manual=viab_preu_solar_manual, intereses_hipoteca=total_intereses,
    )

    if estatic["solar1"] < 0:
        st.error("El cost del sòl surt negatiu amb la rendibilitat objectiu triada. Redueix el percentatge de rendibilitat.")

    st.markdown("")
    st.markdown('<div id="viab-estatic" class="viab-anchor"></div>', unsafe_allow_html=True)
    st.subheader("ANÀLISI ESTÀTIC — COMPTE DE RESULTATS")
    left, right = st.columns((1, 1))
    with left:
        st.markdown("**GASTOS**")
        st_metric(label="Sòl (+ altres costes del sòl)", value=f"{estatic['total_solar']:,.0f} €")
        st_metric(label="Edificació (+ honoraris, llicències, gastos legals, altres)", value=f"{estatic['total_edificacion']:,.0f} €")
        st_metric(label="Administració de la promoció", value=f"{estatic['admin1']:,.0f} €")
        st_metric(label="Comercialització", value=f"{estatic['admin2']:,.0f} €")
        st_metric(label="**TOTAL GASTOS**", value=f"{estatic['total_gastos']:,.0f} €")
    with right:
        st.markdown("**INGRESSOS I RESULTAT**")
        st_metric(label="Ingressos per vendes", value=f"{estatic['ingresos']:,.0f} €")
        st_metric(label="**BAII** (abans d'impostos i interessos)", value=f"{estatic['baii']:,.0f} €")
        st_metric(label="Interessos hipoteca", value=f"{total_intereses:,.0f} €")
        st_metric(label="Gastos de constitució", value=f"{estatic['gastos_constitucio']:,.0f} €")
        st_metric(label="**BAI** (abans d'impostos)", value=f"{estatic['bai']:,.0f} €")

    st.markdown("")
    st.markdown('<div id="viab-dinamic" class="viab-anchor"></div>', unsafe_allow_html=True)
    st.subheader("ANÀLISI DINÀMIC — CASH FLOWS TRIMESTRALS")
    dinamic_display = dinamic_df.copy()
    dinamic_display["TOTAL"] = dinamic_display.sum(axis=1)
    st.markdown(taula_html_es(dinamic_display.round(0), precision=0), unsafe_allow_html=True)
    st.markdown(filedownload(dinamic_display, "Viabilitat_cashflows.xlsx"), unsafe_allow_html=True)

    left, right = st.columns((1, 1))
    with left:
        _viab_fig1 = go.Figure()
        _viab_fig1.add_trace(go.Bar(x=quarters, y=dinamic_df.loc["CASH FLOW ANTES FINANCIACIÓN", quarters], name="Abans de finançament", marker=dict(color=PLOTLY_PALETTE[0])))
        _viab_fig1.add_trace(go.Scatter(x=quarters, y=dinamic_df.loc["CASH FLOW ANTES FINANCIACIÓN ACUM", quarters], name="Acumulat", mode="lines+markers", line=dict(color=PLOTLY_PALETTE[1])))
        _viab_fig1.update_layout(_plotly_layout("Cash flow abans de finançament", "€", title_x="Trimestre"))
        st_plotly_chart(_viab_fig1, use_container_width=True, responsive=True)
    with right:
        _viab_fig2 = go.Figure()
        _viab_fig2.add_trace(go.Bar(x=quarters, y=dinamic_df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN", quarters], name="Després de finançament", marker=dict(color=PLOTLY_PALETTE[0])))
        _viab_fig2.add_trace(go.Scatter(x=quarters, y=dinamic_df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN ACUM", quarters], name="Acumulat", mode="lines+markers", line=dict(color=PLOTLY_PALETTE[1])))
        _viab_fig2.update_layout(_plotly_layout("Cash flow després de finançament", "€", title_x="Trimestre"))
        st_plotly_chart(_viab_fig2, use_container_width=True, responsive=True)

    st.markdown("")
    st.markdown('<div id="viab-resum" class="viab-anchor"></div>', unsafe_allow_html=True)
    st.subheader("RESUM DE RESULTATS")
    _viab_cf_despues = dinamic_df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN", quarters]
    _viab_cf_despues_acum = dinamic_df.loc["CASH FLOW DESPUÉS DE FINANCIACIÓN ACUM", quarters]
    _viab_tir = _viab_calcula_tir(_viab_cf_despues.values)
    _viab_payback = _viab_calcula_payback(_viab_cf_despues_acum)
    _viab_roe = (estatic["bai"] / estatic["recursos_propis"]) * 100 if estatic["recursos_propis"] else np.nan
    _viab_roi = (estatic["baii"] / estatic["total_gastos"]) * 100 if estatic["total_gastos"] else np.nan

    left, center, right = st.columns((1, 1, 1))
    with left:
        st_metric(label="**BAI**", value=f"{estatic['bai']:,.0f} €")
        st_metric(label="**ROE** (retorn recursos propis)", value=f"{_viab_roe:.1f}%" if pd.notna(_viab_roe) else "No disponible")
    with center:
        st_metric(label="**ROI** (retorn de la inversió)", value=f"{_viab_roi:.1f}%" if pd.notna(_viab_roi) else "No disponible")
        st_metric(label="**TIR** anualitzada", value=f"{_viab_tir:.1f}%" if pd.notna(_viab_tir) else "No disponible")
    with right:
        st_metric(label="**Payback**", value=str(_viab_payback) if _viab_payback else "No s'assoleix en el període")

    _viab_inputs_df = pd.DataFrame({
        "Camp": [
            "Municipi del solar", "Superfície construïda (m²)", "Data d'inici de l'operació",
            "Tipus d'interès (%)", "Cost mitjà del m² construït - BEC (€)", f"Preu de venda per m² ({viab_mun}) (€)",
            "Mètode de càlcul del sòl", "Rendibilitat objectiu (%)", "Cost del sòl fixat manualment (€)",
        ],
        "Valor": [
            viab_mun, viab_superficie, str(viab_data_inici),
            viab_tipo_interes, viab_costem2, viab_preciom2,
            viab_metode, viab_rendibilitat, viab_preu_solar_manual,
        ],
    })
    _viab_estatic_df = pd.DataFrame({
        "Concepte": [
            "Sòl (+ altres costes del sòl)", "Edificació (+ honoraris, llicències, gastos legals, altres)",
            "Administració de la promoció", "Comercialització", "TOTAL GASTOS",
            "Ingressos per vendes", "BAII (abans d'impostos i interessos)", "Interessos hipoteca",
            "Gastos de constitució", "BAI (abans d'impostos)",
        ],
        "Import (€)": [
            estatic["total_solar"], estatic["total_edificacion"], estatic["admin1"], estatic["admin2"], estatic["total_gastos"],
            estatic["ingresos"], estatic["baii"], total_intereses, estatic["gastos_constitucio"], estatic["bai"],
        ],
    })
    _viab_resum_df = pd.DataFrame({
        "Indicador": ["BAI (€)", "ROE - retorn recursos propis (%)", "ROI - retorn de la inversió (%)", "TIR anualitzada (%)", "Payback"],
        "Valor": [estatic["bai"], _viab_roe, _viab_roi, _viab_tir, str(_viab_payback) if _viab_payback else "No s'assoleix en el període"],
    })

    def _viab_build_resum_excel():
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite) as writer:
            _viab_inputs_df.to_excel(writer, sheet_name="Dades d'entrada", index=False, header=True)
            _viab_curves_edited.to_excel(writer, sheet_name="Corbes trimestrals (%)", index=True, header=True)
            _viab_estatic_df.to_excel(writer, sheet_name="Anàlisi estàtic", index=False, header=True)
            dinamic_display.to_excel(writer, sheet_name="Anàlisi dinàmic", index=True, header=True)
            _viab_resum_df.to_excel(writer, sheet_name="Resum de resultats", index=False, header=True)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode("latin-1")
        return f'''<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Viabilitat_resum_complet.xlsx">
        <button class="download-button">Descarregar resum complet (Excel)</button></a>'''

    st.markdown("")
    st.markdown(_viab_build_resum_excel(), unsafe_allow_html=True)

############################################################ ESTUDI D'OFERTA D'OBRA NOVA APCE (afegit) ############################################################
# Integració autocontinguda de Z:\ESTUDIS\APP\Estudi-oferta\Estudi_oferta_atlas.py com a nova
# pestanya. Tot aquest bloc porta el prefix "oferta_" per blindar-lo de qualsevol col·lisió de
# nom amb la resta del fitxer -- en especial filedownload, load_css_file i load_shp, que
# existeixen A L'ALTRE fitxer amb una implementació diferent i que NO es reutilitzen ni se
# sobreescriuen aquí (es creen versions pròpies, independents, amb la lògica de l'app de
# referència). No es toca ni es modifica res del que ja existia abans d'aquest bloc: només
# s'hi afegeix codi nou.
# Es reutilitzen expressament (no hi ha col·lisió i mantenen una única font de veritat):
# DATA_FILE_ATLAS_OFERTA, ATLAS_PERIODES, SHAPEFILE_MUN, PLOTLY_PALETTE, CSS_COLORS,
# taula_html_es, _es_num_str, st_plotly_chart, i el switch de tema clar/fosc ja existent
# (st.session_state["theme"]) -- NO s'afegeix un segon selector de tema.

OFERTA_PAGINES = ["Catalunya", "Províncies i àmbits", "Municipis", "Districtes de Barcelona", "Mapa interactiu"]
OFERTA_EDICIONS = ["2025", "2026"]
OFERTA_NOM_FITXER_XLSX = "BBDD_Atlas_trimmed.xlsx"  # fallback si encara no existeix el .json

# Paleta pròpia: adaptació "rol per rol" de la paleta verda de l'app original als colors
# ja existents en aquesta app (PLOTLY_PALETTE / CSS_COLORS), sense importar cap color nou.
OFERTA_COLOR_BARRES = PLOTLY_PALETTE[0]    # blau: color principal de barres (mateix criteri que bar_plotly/line_plotly)
OFERTA_COLOR_ACCENT = PLOTLY_PALETTE[1]    # taronja APCE: accent / segona sèrie
OFERTA_COLOR_CLAR = CSS_COLORS["accent"]   # taronja clar: extrem "baix" dels degradats (mateixa parella accent/primary del tema)
OFERTA_COLOR_FOSC = CSS_COLORS["primary"]  # taronja fosc: extrem "alt" dels degradats
OFERTA_COLOR_MUTED = PLOTLY_PALETTE[3]     # gris: contorns/textos secundaris
OFERTA_COLOR_SURFACE = "#fff7ed"           # superfície clara (LIGHT_THEME["surface-solid"])
OFERTA_COLOR_FONS_GRAFIC = "rgba(0, 0, 0, 0)"  # fons transparent, mateix criteri que _plotly_layout

OFERTA_TITOL_AMBIT = {
    "Metropolità": "Àmbit metropolità",
    "Comarques Gironines": "Àmbit de les Comarques Gironines",
    "Penedès": "Àmbit del Penedès",
    "Camp de Tarragona": "Àmbit del Camp de Tarragona",
    "Alt Pirineu i Aran": "Àmbit de l'Alt Pirineu i Aran",
    "Ponent": "Àmbit de Ponent",
    "Comarques Centrals": "Àmbit de les Comarques Centrals",
    "Terres de l'Ebre": "Àmbit de les Terres de l'Ebre",
}

OFERTA_NOTA_MEDIANA = (
    "En els histogrames, la línia discontínua marca la **mediana**: el valor central, "
    "que deixa la meitat dels habitatges per sota i l'altra meitat per sobre. A diferència "
    "de la mitjana, la mediana no es distorsiona pels preus molt alts o molt baixos."
)

OFERTA_TEXTOS_INFORME_2026 = {
    "introduccio": """L'anàlisi s'estructura en nivells territorials de detall creixent: resultats agregats per al conjunt de Catalunya, desglossament per províncies, per àmbits territorials, per corones metropolitanes de Barcelona —analitzant de manera diferenciada el municipi de Barcelona, la primera i la segona corona—, i finalment a escala municipal i de districte de la ciutat de Barcelona, identificant els mercats locals amb major oferta i els nivells de preu de cada territori.""",
    "tipologies": """L'oferta continua dominada per l'habitatge plurifamiliar i concentrada en les tipologies de 2 i 3 dormitoris, mentre el producte compacte registra els preus unitaris més elevats.""",
    "territori": """El diferencial territorial al llarg de Catalunya s'amplia: davant la pressió del litoral i l'entorn metropolità, el Camp de Tarragona es manté estable i els àmbits d'interior — Ponent, Comarques Centrals i Terres de l'Ebre— conserven preus a l'entorn o per sota dels 3.600 €/m² i mercats d'obra nova encara amb volums relativament limitats, amb l'Alt Pirineu i Aran com a excepció singular lligada a la segona residència de muntanya amb un producte més orientat al segment premium.""",
    "provincies": """El mapa provincial confirma la forta heterogeneïtat territorial del mercat d'obra nova a Catalunya. Barcelona (5.513 €/m²) i Girona (5.397 €/m²) es consoliden com les províncies de preus més alts en nivells per damunt de la mitjana catalana (4.657 €/m²), mentre que Tarragona (3.558 €/m²) i Lleida (3.048 €/m²) mantenen xifres sensiblement inferiors. L'evolució interanual accentua aquesta bretxa: Girona registra el major creixement (+14,9% en preu/m²), seguida de Barcelona (+8,2%) i Tarragona (+7,5%), amb Lleida pràcticament estable (+0,4%). Barcelona concentra, a més, dues de cada tres promocions actives a Catalunya, la qual cosa evidencia que la pressió de demanda i l'escassetat de producte continuen focalitzades a l'eix metropolità i el litoral nord.""",
    "ambits": """L'anàlisi per àmbits territorials revela dos pols de preu diferenciats. Al capdavant se situen el Metropolità (5.631 €/m²), que concentra el gruix de l'activitat promotora. El segueixen les Comarques Gironines (5.345 €/m²) i el Penedès (5.149 €/m²), que es beneficien de la seva condició de territoris d'expansió natural de l'àrea metropolitana i del litoral nord, i, de manera destacada, l'Alt Pirineu i Aran (5.079 €/m²), on el producte de muntanya orientat a segona residència assoleix preus unitaris comparables als metropolitans malgrat la seva reduïda dimensió de mercat. En un segon esglaó se situen el Camp de Tarragona (3.932 €/m²) i les Comarques Centrals (3.564 €/m²), mentre que Ponent (2.831 €/m²) i les Terres de l'Ebre (3.074 €/m²) tanquen la classificació amb els preus més assequibles de Catalunya.""",
    "districtes_barcelona": """L'anàlisi per districtes revela una Barcelona a dues velocitats, amb un diferencial important entre extrems. L'Eixample (17.594 €/m²) i Sant Martí (12.317 €/m²), únics districtes per damunt de la mitjana municipal, concentren una oferta escassa i de perfil luxe dirigida a un comprador patrimonialista i internacional, la qual cosa dispara la mitjana d'ambdós districtes. Per sota de la mitjana de la ciutat s'ordenen Horta-Guinardó (8.596 €/m²), Gràcia (7.577 €/m²), Sant Andreu (7.390 €/m²), Ciutat Vella (6.664 €/m²) i Sants-Montjuïc (6.335 €/m²), si bé algunes d'aquestes xifres vénen definides pel poc producte puntualment en oferta durant el primer semestre de l'any.""",
}


def oferta_mostra_text_informe(clau, any_estudi):
    if str(any_estudi) != "2026":
        return
    text = OFERTA_TEXTOS_INFORME_2026.get(clau)
    if text:
        st.markdown(f'<div class="oferta-text-informe">{text}</div>', unsafe_allow_html=True)


OFERTA_VARIABLES_QUALITATS = [
    "Aire condicionat", "Bomba de calor", "Aerotèrmia", "Calefacció",
    "Preinstal·lació d'A.C./B. Calor/Calefacció", "Parquet", "Armaris encastats",
    "Placa de cocció amb gas", "Placa de cocció vitroceràmica", "Placa d'inducció", "Plaques solars",
]
OFERTA_VARIABLES_EQUIPAMENTS = [
    "Zona enjardinada", "Parc infantil", "Piscina comunitària", "Traster", "Ascensor",
    "Equipament Esportiu", "Sala de jocs", "Sauna", "Altres", "Cap dels anteriors",
]
OFERTA_VARIABLES_CARACTERISTIQUES = [
    "Total dormitoris", "Banys i lavabos", "Cuines estàndard", "Cuines americanes",
    "Terrasses, balcons i patis", "Estudi/golfes", "Safareig", "Altres interiors", "Altres exteriors",
]

OFERTA_COLUMNES_NECESSARIES = [
    "period_id", "ID", "municipality", "province", "municipality_id", "nom_amb", "corona",
    "bcn_district", "price", "price_m2_util", "useful_size", "bedrooms", "bathrooms",
    "property_type", "energy_certification_type", "heating_type", "latitude", "longitude",
    "lift", "terrace", "garage", "storage", "swimming_pool", "garden", "air_conditioning",
    "heating_individual", "heating_central", "fitted_wardrobes",
]


def oferta_netejar_text(serie):
    return serie.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def oferta_categoritzar_dormitoris(valor):
    if pd.isna(valor):
        return pd.NA
    valor = int(valor)
    return "4+D" if valor >= 4 else f"{valor}D"


def oferta_categoritzar_banys(valor):
    if pd.isna(valor):
        return pd.NA
    valor = int(valor)
    return f"{valor} Bany" if valor <= 1 else "2 i més Banys"


def oferta_normalitzar_certificat_energetic(valor):
    if pd.isna(valor):
        return "Sense informació"
    valor = str(valor).strip().upper()
    if valor in ["A", "B", "C", "D", "E", "F", "G"]:
        return valor
    if "TRAM" in valor or "PROCESS" in valor:
        return "En tràmits"
    return "Sense informació"


def oferta_validar_columnes(df):
    faltants = [col for col in OFERTA_COLUMNES_NECESSARIES if col not in df.columns]
    if faltants:
        raise ValueError("Falten columnes necessàries a la base de dades: " + ", ".join(faltants))


def oferta_normalitzar_dades(df):
    df = df.copy()
    oferta_validar_columnes(df)
    df = df[df["period_id"].isin(ATLAS_PERIODES)].copy()

    columnes_text = ["period_id", "municipality", "province", "nom_amb", "corona", "bcn_district", "property_type", "energy_certification_type", "heating_type"]
    columnes_num = ["ID", "municipality_id", "price", "price_m2_util", "useful_size", "bedrooms", "bathrooms", "latitude", "longitude", "lift", "terrace", "garage", "storage", "swimming_pool", "garden", "air_conditioning", "heating_individual", "heating_central", "fitted_wardrobes"]

    for col in columnes_text:
        df[col] = oferta_netejar_text(df[col])
    for col in columnes_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["lift", "terrace", "garage", "storage", "swimming_pool", "garden", "air_conditioning", "heating_individual", "heating_central", "fitted_wardrobes"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


def oferta_adaptar_dades(df):
    df = oferta_normalitzar_dades(df)

    df["Any"] = pd.to_numeric(df["any"], errors="coerce").astype("Int64")
    df["Municipi"] = df["municipality"]
    df["PROVINCIA"] = df["province"]
    df["TERRITORI"] = df["nom_amb"]
    df["COD_Nom_Corona"] = df["corona"]
    df["Nom DIST"] = df["bcn_district"]
    df["CODIMUN"] = df["municipality_id"]

    df["Preu mitjà"] = df["price"]
    df["Preu m2 útil"] = df["price_m2_util"]
    df["Superfície útil"] = df["useful_size"]
    df["Total dormitoris"] = df["bedrooms"]
    df["Banys i lavabos"] = df["bathrooms"]
    df["Total dormitoris_cat"] = df["bedrooms"].apply(oferta_categoritzar_dormitoris)
    df["Banys i lavabos_cat"] = df["bathrooms"].apply(oferta_categoritzar_banys)

    df["TIPOG"] = np.where(
        df["clase_vivienda"].astype("string").str.lower().eq("unifamiliar"),
        "Habitatges unifamiliars", "Habitatges plurifamiliars",
    )
    df["TIPO"] = df["TIPOG"]
    df["TIPH"] = "De nova Construcció"
    df["QENERGC"] = df["energy_certification_type"].apply(oferta_normalitzar_certificat_energetic)

    df["Zona enjardinada"] = df["garden"]
    df["Piscina comunitària"] = df["swimming_pool"]
    df["Traster"] = df["storage"]
    df["Ascensor"] = df["lift"]
    df["Terrasses, balcons i patis"] = df["terrace"]
    df["Equipament Esportiu"] = df["sports"]

    df["Aire condicionat"] = df["air_conditioning"]
    df["Armaris encastats"] = df["fitted_wardrobes"]
    df["Calefacció"] = ((df["heating_individual"] == 1) | (df["heating_central"] == 1)).astype(int)

    ht = df["heating_type"].astype("string").str.lower().fillna("")
    df["De gasoil"] = ht.str.contains("gasoil|gasóleo|gasoleo").astype(int)
    df["De propà"] = ht.str.contains("propan|propà|butan").astype(int)
    df["De gas natural"] = (ht.str.contains("natural") | ht.eq("gas")).astype(int)
    df["D'electricitat"] = ht.str.contains("el[eé]ctr|bomba").astype(int)
    df["Bomba de calor"] = ht.str.contains("bomba").astype(int)
    tipus_coneguts = df[["De gasoil", "De propà", "De gas natural", "D'electricitat"]].sum(axis=1)
    df["No s'indica tipus"] = (tipus_coneguts == 0).astype(int)

    df["APAR"] = np.where(df["garage"] == 1, "Amb plaça d'aparcament", "Sense informació")

    for col in ["Parc infantil", "Sala de jocs", "Sauna", "Altres", "Cap dels anteriors", "Aerotèrmia", "Preinstal·lació d'A.C./B. Calor/Calefacció", "Parquet", "Placa de cocció amb gas", "Placa de cocció vitroceràmica", "Placa d'inducció", "Plaques solars", "Cuines estàndard", "Cuines americanes", "Estudi/golfes", "Safareig", "Altres interiors", "Altres exteriors"]:
        df[col] = 0

    return df


@st.cache_data(show_spinner="Carregant les dades de l'Estudi d'oferta...")
def oferta_carregant_dades():
    """Prioritza el JSON (molt més ràpid de llegir que l'Excel); si no existeix, Excel."""
    if Path(DATA_FILE_ATLAS_OFERTA).exists():
        df = pd.read_json(DATA_FILE_ATLAS_OFERTA, orient="records")
    elif Path(OFERTA_NOM_FITXER_XLSX).exists():
        df = pd.read_excel(OFERTA_NOM_FITXER_XLSX, sheet_name=0)
    else:
        raise FileNotFoundError(f"No es troba {DATA_FILE_ATLAS_OFERTA} ni {OFERTA_NOM_FITXER_XLSX}.")
    return oferta_adaptar_dades(df)


@st.cache_data(show_spinner=False)
def oferta_crear_bases_any(df, any_estudi):
    return df[df["Any"] == int(any_estudi)].copy()


def oferta_dades_any(dades_2025, dades_2026, selected_edition):
    return dades_2025 if str(selected_edition) == "2025" else dades_2026


def oferta_preparar_fig(fig):
    fig.update_layout(
        paper_bgcolor=OFERTA_COLOR_FONS_GRAFIC,
        plot_bgcolor=OFERTA_COLOR_FONS_GRAFIC,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    # st_plotly_chart (reutilitzat per pintar aquests gràfics) fixa title.font sense
    # title.text; si la figura no té cap títol propi, Plotly mostra literalment
    # "undefined". La majoria d'aquests gràfics no en porten (el títol el posa el
    # markdown de sobre, via oferta_mostra), així que aquí es garanteix un text buit.
    if fig.layout.title is None or fig.layout.title.text is None:
        fig.update_layout(title=dict(text=""))
    return fig


def oferta_mostra(titol, figura):
    st.markdown(f"**{titol}**")
    st_plotly_chart(figura, use_container_width=True, responsive=True)


def oferta_titol_seccio(text):
    st.subheader(text)


def oferta_fig_no_disponible(titulo, motivo):
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{titulo}</b><br>{motivo}",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(size=14), align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=350)
    return oferta_preparar_fig(fig)


def oferta_mitjana(s):
    return pd.to_numeric(s, errors="coerce").mean()


def oferta_pct(value, total):
    if total in [0, None] or pd.isna(total):
        return pd.NA
    return value * 100 / total


def oferta_format_num(x, decimals=1):
    if pd.isna(x):
        return "n.d."
    return _es_num_str(f"{x:,.{decimals}f}")


def oferta_format_catala(x):
    if pd.isna(x):
        return "n.d."
    return _es_num_str(f"{x:,.0f}")


def oferta_resum_basic(df_hab):
    if df_hab.empty:
        return {"unitats": 0, "preu": pd.NA, "preum2": pd.NA, "superficie": pd.NA}
    return {
        "unitats": len(df_hab),
        "preu": oferta_mitjana(df_hab["Preu mitjà"]),
        "preum2": oferta_mitjana(df_hab["Preu m2 útil"]),
        "superficie": oferta_mitjana(df_hab["Superfície útil"]),
    }


def oferta_text_resum_cat(df_hab, any_estudi):
    r = oferta_resum_basic(df_hab)
    municipis = df_hab["Municipi"].nunique()
    top_mun = df_hab["Municipi"].value_counts().head(5).index.tolist()
    top_mun_txt = ", ".join(top_mun) if top_mun else "n.d."
    return f"""
    <p style="margin-top: 10px">
    L'Estudi d'Oferta de Nova Construcció per a {any_estudi} inclou {oferta_format_num(r['unitats'],0)} habitatges únics en oferta, distribuïts en {oferta_format_num(municipis,0)} municipis de Catalunya.
    El preu mitjà dels habitatges és de {oferta_format_num(r['preu'],0)} €, el preu mitjà per m² útil és de {oferta_format_num(r['preum2'],0)} €/m² i la superfície útil mitjana se situa en {oferta_format_num(r['superficie'],1)} m².
    Els municipis amb més habitatges detectats en la mostra són: {top_mun_txt}.
    </p>
    """


def oferta_text_resum_geo(df_hab, geo, columna_geo, any_estudi):
    df = df_hab[df_hab[columna_geo] == geo].copy()
    r = oferta_resum_basic(df)
    total = len(df_hab)
    pes = oferta_pct(len(df), total)
    return f"""
    Els resultats de l'Estudi d'Oferta de nova construcció de {any_estudi} per a {geo} mostren {oferta_format_num(r['unitats'],0)} habitatges en oferta, que representen el {oferta_format_num(pes,1)}% de la mostra filtrada.
    El preu mitjà dels habitatges en venda és de {oferta_format_num(r['preu'],0)} €, amb una superfície útil mitjana de {oferta_format_num(r['superficie'],1)} m².
    Per tant, el preu per m² útil se situa en {oferta_format_num(r['preum2'],0)} €/m² de mitjana.
    """


def oferta_text_resum_mun_dis(df_hab, geo, columna_geo, any_estudi):
    df = df_hab[df_hab[columna_geo] == geo].copy()
    r = oferta_resum_basic(df)
    plur = (df["TIPOG"] == "Habitatges plurifamiliars").sum()
    unif = (df["TIPOG"] == "Habitatges unifamiliars").sum()
    return f"""
    <p>
    L'any {any_estudi}, {geo} registra {oferta_format_num(r['unitats'],0)} habitatges únics en oferta.
    La superfície útil mitjana és de {oferta_format_num(r['superficie'],1)} m², el preu mitjà és de {oferta_format_num(r['preu'],0)} € i el preu per m² útil és de {oferta_format_num(r['preum2'],0)} €/m².
    La mostra inclou {oferta_format_num(plur,0)} habitatges plurifamiliars i {oferta_format_num(unif,0)} habitatges unifamiliars.
    </p>
    """


def oferta_filedownload(df, filename):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    return f'''<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>'''


@st.cache_data(show_spinner=False)
def oferta_construir_df_final(df):
    filas = []
    variables = [
        ("Unitats", "Unitats"),
        ("Superfície mitjana (m² útils)", "m² útils"),
        ("Preu mitjà de venda de l'habitatge (€)", "€"),
        ("Preu de venda per m² útil (€)", "€/m² útil"),
    ]
    nivells = [
        ("Catalunya", lambda d: pd.Series("Catalunya", index=d.index), None),
        ("Províncies", lambda d: d["PROVINCIA"], None),
        ("Àmbits territorials", lambda d: d["TERRITORI"], None),
        ("Municipis", lambda d: d["Municipi"], "CODIMUN"),
        ("Districtes de Barcelona", lambda d: d["Nom DIST"], None),
    ]
    tipologies = {
        "TOTAL HABITATGES": None,
        "HABITATGES PLURIFAMILIARS": "Habitatges plurifamiliars",
        "HABITATGES UNIFAMILIARS": "Habitatges unifamiliars",
    }
    for any_estudi in sorted(df["Any"].dropna().unique()):
        d_any = df[df["Any"] == any_estudi].copy()
        for nom_nivell, func_geo, col_codi in nivells:
            d_any["_GEO"] = func_geo(d_any)
            if nom_nivell == "Districtes de Barcelona":
                d_nivell_base = d_any[(d_any["Municipi"] == "Barcelona") & d_any["_GEO"].notna()].copy()
            else:
                d_nivell_base = d_any[d_any["_GEO"].notna()].copy()
            for tipologia_label, tipologia_filtre in tipologies.items():
                d_tipus = d_nivell_base.copy() if tipologia_filtre is None else d_nivell_base[d_nivell_base["TIPOG"] == tipologia_filtre].copy()
                if d_tipus.empty:
                    continue
                for geo, grup in d_tipus.groupby("_GEO", dropna=False):
                    codiine = pd.NA
                    if col_codi is not None and col_codi in grup.columns:
                        valors_codi = grup[col_codi].dropna().unique()
                        codiine = valors_codi[0] if len(valors_codi) else pd.NA
                    valors = {
                        "Unitats": len(grup),
                        "Superfície mitjana (m² útils)": oferta_mitjana(grup["Superfície útil"]),
                        "Preu mitjà de venda de l'habitatge (€)": oferta_mitjana(grup["Preu mitjà"]),
                        "Preu de venda per m² útil (€)": oferta_mitjana(grup["Preu m2 útil"]),
                    }
                    for variable, unitats in variables:
                        filas.append({
                            "Any": int(any_estudi), "Nivell": nom_nivell, "GEO": geo,
                            "Tipologia": tipologia_label, "Variable": variable,
                            "Valor": valors[variable], "Unitats": unitats, "codiine": codiine,
                        })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def oferta_grafic_provincies(df_prom):
    provprom_map = df_prom[["PROVINCIA"]].value_counts().reset_index()
    provprom_map.columns = ["PROVINCIA", "Habitatges"]
    fig = px.bar(provprom_map.sort_values("Habitatges"), x="Habitatges", y="PROVINCIA", orientation="h", labels={"PROVINCIA": "Província", "Habitatges": "Habitatges en oferta"})
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_top_municipis(df_hab, top=20):
    dades = df_hab["Municipi"].value_counts().head(top).sort_values().reset_index()
    dades.columns = ["Municipi", "Habitatges en oferta"]
    fig = px.bar(dades, x="Habitatges en oferta", y="Municipi", orientation="h")
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


def _oferta_codi_provincia(codi_municipi):
    if pd.isna(codi_municipi):
        return pd.NA
    return str(int(codi_municipi)).zfill(5)[:2]


def _oferta_nom_provincia(codi_municipi):
    return {"08": "Barcelona", "17": "Girona", "25": "Lleida", "43": "Tarragona"}.get(_oferta_codi_provincia(codi_municipi), pd.NA)


def _oferta_fig_mapa_oferta(gdf, locations, z, customdata, hovertemplate, zoom=6.35):
    fig = go.Figure(go.Choroplethmapbox(
        geojson=gdf.__geo_interface__,
        locations=locations,
        z=z,
        featureidkey="properties._map_id",
        colorscale=[[0, OFERTA_COLOR_CLAR], [1, OFERTA_COLOR_FOSC]],
        marker_opacity=0.82,
        marker_line_width=0.35,
        customdata=customdata,
        hovertemplate=hovertemplate,
        zmin=0,
    ))
    fig.update_layout(
        title=dict(text=""),  # evita que st_plotly_chart (que fixa title.font sense text) mostri "undefined"
        mapbox_style="carto-positron",
        mapbox_zoom=zoom,
        mapbox_center={"lat": 41.65, "lon": 1.55},
        margin=dict(l=0, r=0, t=0, b=0),
        height=440,
    )
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_mapa_municipal_oferta(df_hab, _shp):
    if _shp is None:
        return oferta_fig_no_disponible("Mapa municipal", "No s'ha trobat la geometria municipal.")
    shp = _shp.copy()
    shp["_map_id"] = shp["municipi"].astype(str)
    recompte = df_hab.groupby("CODIMUN").size().reset_index(name="Habitatges")
    recompte["municipi"] = pd.to_numeric(recompte["CODIMUN"], errors="coerce").astype("Int64")
    mapa = shp.merge(recompte[["municipi", "Habitatges"]], on="municipi", how="left")
    mapa["Habitatges"] = pd.to_numeric(mapa["Habitatges"], errors="coerce")
    mapa["Habitatges_txt"] = mapa["Habitatges"].map(lambda x: oferta_format_catala(x) if pd.notna(x) else "")
    customdata = np.stack([mapa["nom_muni"].fillna("n.d."), mapa["Habitatges_txt"]], axis=-1)
    return _oferta_fig_mapa_oferta(mapa, locations=mapa["_map_id"], z=mapa["Habitatges"], customdata=customdata, hovertemplate="<b>%{customdata[0]}</b><br>Habitatges en oferta: %{customdata[1]}<extra></extra>", zoom=6.35)


@st.cache_data(show_spinner=False)
def oferta_mapa_provincial_oferta(df_hab, _shp):
    if _shp is None:
        return oferta_fig_no_disponible("Mapa provincial", "No s'ha trobat la geometria municipal.")
    shp = _shp.copy()
    shp["PROVINCIA"] = shp["municipi"].apply(_oferta_nom_provincia)
    shp = shp.dropna(subset=["PROVINCIA"])
    prov = shp.dissolve(by="PROVINCIA", as_index=False)[["PROVINCIA", "geometry"]]
    prov["_map_id"] = prov["PROVINCIA"]
    recompte = df_hab["PROVINCIA"].value_counts().reset_index()
    recompte.columns = ["PROVINCIA", "Habitatges"]
    mapa = prov.merge(recompte, on="PROVINCIA", how="left")
    mapa["Habitatges"] = pd.to_numeric(mapa["Habitatges"], errors="coerce")
    mapa["Habitatges_txt"] = mapa["Habitatges"].map(lambda x: oferta_format_catala(x) if pd.notna(x) else "")
    customdata = np.stack([mapa["PROVINCIA"], mapa["Habitatges_txt"]], axis=-1)
    return _oferta_fig_mapa_oferta(mapa, locations=mapa["_map_id"], z=mapa["Habitatges"], customdata=customdata, hovertemplate="<b>%{customdata[0]}</b><br>Habitatges en oferta: %{customdata[1]}<extra></extra>", zoom=6.15)


@st.cache_data(show_spinner=False)
def oferta_grafic_principals_tipologies(df_hab):
    if df_hab.empty:
        return oferta_fig_no_disponible("Característiques", "No hi ha dades per al filtre seleccionat.")
    taula = df_hab.groupby(["Total dormitoris", "Banys i lavabos"]).size().div(len(df_hab)).reset_index(name="Proporcions").sort_values(by="Proporcions", ascending=False)
    taula["Proporcions"] = taula["Proporcions"] * 100
    taula["Tipologia"] = np.where(
        taula["Banys i lavabos"].fillna(0).astype(int) == 1,
        taula["Total dormitoris"].fillna(0).astype(int).astype(str) + " dormitoris i " + taula["Banys i lavabos"].fillna(0).astype(int).astype(str) + " bany",
        taula["Total dormitoris"].fillna(0).astype(int).astype(str) + " dormitoris i " + taula["Banys i lavabos"].fillna(0).astype(int).astype(str) + " banys",
    )
    fig = px.bar(taula.head(4), x="Proporcions", y="Tipologia", orientation="h", title="Principals tipologies dels habitatges en oferta (%)")
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_obra_nova(df_hab):
    taula = df_hab[["TIPH"]].value_counts().reset_index()
    taula.columns = ["Tipus", "Habitatges en oferta"]
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=taula["Tipus"], values=taula["Habitatges en oferta"], hole=0.5, showlegend=True, marker=dict(colors=[OFERTA_COLOR_BARRES, OFERTA_COLOR_ACCENT]), textposition="outside", textinfo="percent+label"))
    fig.update_layout(title="Habitatges en oferta d'obra nova a Catalunya (%)")
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_qualitats(df_hab):
    disponibles = [c for c in OFERTA_VARIABLES_QUALITATS if c in df_hab.columns]
    if not disponibles or df_hab.empty:
        return oferta_fig_no_disponible("Qualitats", "No hi ha variables de qualitats disponibles.")
    taula = df_hab[disponibles].sum(axis=0, numeric_only=True)
    taula = pd.DataFrame({"Qualitats": taula.index, "Total": taula.values})
    taula["Total"] = taula["Total"] * 100 / len(df_hab)
    taula = taula.sort_values("Total", ascending=True)
    fig = px.bar(taula, x="Total", y="Qualitats", orientation="h", labels={"Total": "Proporcions sobre el total d'habitatges (%)"})
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_equipaments(df_hab):
    disponibles = [c for c in OFERTA_VARIABLES_EQUIPAMENTS if c in df_hab.columns]
    if not disponibles or df_hab.empty:
        return oferta_fig_no_disponible("Equipaments", "No hi ha variables d'equipaments disponibles.")
    taula = df_hab[disponibles].sum(axis=0, numeric_only=True)
    taula = pd.DataFrame({"Equipaments": taula.index, "Total": taula.values})
    taula["Total"] = taula["Total"] * 100 / len(df_hab)
    taula = taula.sort_values("Total", ascending=True)
    fig = px.bar(taula, x="Total", y="Equipaments", orientation="h", labels={"Total": "Proporcions sobre el total d'habitatges (%)"})
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_qualitats_equipaments(df_hab):
    if df_hab.empty:
        return oferta_fig_no_disponible("Qualitats i equipaments", "No hi ha dades.")
    files = []
    for grup, variables in [("Qualitats", OFERTA_VARIABLES_QUALITATS), ("Equipaments", OFERTA_VARIABLES_EQUIPAMENTS)]:
        for variable in variables:
            if variable in df_hab.columns:
                percentatge = pd.to_numeric(df_hab[variable], errors="coerce").sum() * 100 / len(df_hab)
                if percentatge > 0:
                    files.append({"Variable": variable, "Grup": grup, "Percentatge": percentatge})
    taula = pd.DataFrame(files)
    if taula.empty:
        return oferta_fig_no_disponible("Qualitats i equipaments", "No hi ha variables disponibles amb valor.")
    taula = taula.sort_values("Percentatge", ascending=True)
    fig = px.bar(taula, x="Percentatge", y="Variable", color="Grup", orientation="h",
                 color_discrete_map={"Qualitats": OFERTA_COLOR_BARRES, "Equipaments": OFERTA_COLOR_ACCENT},
                 labels={"Percentatge": "Habitatges amb aquesta característica (%)", "Variable": ""})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_mitjanes(df_hab, columna, etiqueta):
    if df_hab.empty:
        return oferta_fig_no_disponible(etiqueta, "No hi ha dades.")
    metriques = ["Superfície útil", "Preu mitjà", "Preu m2 útil"]
    per_tipus = df_hab.groupby(["TIPOG", "Total dormitoris_cat"])[metriques].mean(numeric_only=True).reset_index()
    total = df_hab.groupby("Total dormitoris_cat")[metriques].mean(numeric_only=True).reset_index()
    total["TIPOG"] = "Total habitatges"
    taula = pd.concat([per_tipus, total], axis=0).rename(columns={"TIPOG": "Tipologia"})
    fig = px.bar(taula, x=columna, y="Total dormitoris_cat", color="Tipologia", orientation="h",
                 color_discrete_sequence=PLOTLY_PALETTE[:3], barmode="group",
                 labels={columna: etiqueta, "Total dormitoris_cat": "Tipologia d'habitatge"})
    fig.update_layout(font=dict(size=13), legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=0.75))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_variacio_anual(table_any_actual, table_any_anterior):
    if table_any_actual.empty or table_any_anterior.empty:
        return oferta_fig_no_disponible("Comparativa", "No hi ha dades per comparar.")
    actual = table_any_actual.groupby("TIPOG").agg({"Superfície útil": "mean", "Preu mitjà": "mean", "Preu m2 útil": "mean", "ID": "size"}).rename(columns={"ID": "Unitats"})
    anterior = table_any_anterior.groupby("TIPOG").agg({"Superfície útil": "mean", "Preu mitjà": "mean", "Preu m2 útil": "mean", "ID": "size"}).rename(columns={"ID": "Unitats"})
    cols = ["Unitats", "Superfície útil", "Preu mitjà", "Preu m2 útil"]
    variacions = ((actual[cols] / anterior[cols] - 1) * 100).reset_index().melt(id_vars="TIPOG", var_name="Indicador", value_name="Variació")
    fig = px.bar(variacions, x="Indicador", y="Variació", color="TIPOG", barmode="group", color_discrete_sequence=PLOTLY_PALETTE[:3], labels={"Variació": "Variació anual (%)", "TIPOG": "Tipologia"})
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_taula_comparativa(df_final, nivell, geo, any_ini, any_fin):
    df = df_final[(df_final["Nivell"] == nivell) & (df_final["Any"].between(any_ini, any_fin))]
    if geo is not None:
        df = df[df["GEO"] == geo]
    taula = df.pivot_table(index="Any", columns=["Tipologia", "Variable"], values="Valor", aggfunc="first")
    return taula.sort_index(axis=1, level=[0, 1]).round(0)


@st.cache_data(show_spinner=False)
def oferta_grafic_tipologia_donut(df):
    if df.empty:
        return oferta_fig_no_disponible("Tipologia", "No hi ha dades.")
    data = df["TIPOG"].value_counts().reset_index()
    data.columns = ["Tipologia", "Habitatges"]
    fig = px.pie(data, values="Habitatges", names="Tipologia", hole=0.5, color_discrete_sequence=PLOTLY_PALETTE[:3])
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return oferta_preparar_fig(fig)


def oferta_filtra_geo(df, columna_geo, valor):
    if columna_geo is None:
        return df
    if columna_geo == "Nom DIST":
        return df[(df["Municipi"] == "Barcelona") & (df["Nom DIST"] == valor)]
    return df[df[columna_geo] == valor]


@st.cache_data(show_spinner=False)
def oferta_grafic_dormitoris(df):
    recompte = df["Total dormitoris_cat"].value_counts().reset_index()
    recompte.columns = ["Dormitoris", "Habitatges"]
    fig = px.bar(recompte.sort_values("Dormitoris"), x="Dormitoris", y="Habitatges")
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_lavabos(df):
    recompte = df["Banys i lavabos_cat"].value_counts().reset_index()
    recompte.columns = ["Banys i lavabos", "Habitatges"]
    fig = px.bar(recompte.sort_values("Banys i lavabos"), x="Banys i lavabos", y="Habitatges")
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_tipologia_pie(df):
    recompte = df["TIPOG"].value_counts(normalize=True).mul(100).reset_index()
    recompte.columns = ["Tipologia", "Proporció"]
    fig = px.pie(recompte, values="Proporció", names="Tipologia", hole=0.4, color_discrete_sequence=[OFERTA_COLOR_CLAR, OFERTA_COLOR_FOSC])
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_distribucio(df, kpi):
    if df.empty:
        return oferta_fig_no_disponible(kpi, "No hi ha dades per a la zona seleccionada.")
    fig = px.histogram(df, x=kpi, nbins=25)
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    med = pd.to_numeric(df[kpi], errors="coerce").median()
    if pd.notna(med):
        fig.add_vline(x=med, line_dash="dash", line_width=2, line_color=OFERTA_COLOR_ACCENT, annotation_text=f"Mediana: {oferta_format_num(med, 0)}", annotation_position="top")
    fig.update_layout(xaxis_title=kpi, yaxis_title="Nombre d'habitatges")
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_matriu_hab_lav(df, pivot_name):
    cols = ["Preu mitjà", "Preu m2 útil", "Superfície útil"]
    resum = df.groupby(["Total dormitoris", "Banys i lavabos"])[cols].mean().reset_index()
    resum = resum[(resum["Total dormitoris"] > 0) & (resum["Banys i lavabos"] > 0)]
    resum["Total dormitoris"] = resum["Total dormitoris"].astype(int).astype(str) + " habitacions"
    resum["Banys i lavabos"] = resum["Banys i lavabos"].astype(int).astype(str) + " lavabos"
    resum[cols] = resum[cols].map(oferta_format_catala)
    return resum.pivot(index="Total dormitoris", columns="Banys i lavabos", values=pivot_name)


@st.cache_data(show_spinner=False)
def oferta_grafic_caracteristiques(df):
    if df.empty:
        return oferta_fig_no_disponible("Característiques", "No hi ha dades.")
    vals = df[OFERTA_VARIABLES_CARACTERISTIQUES].mean(numeric_only=True).reset_index()
    vals.columns = ["Característica", "Total"]
    fig = px.bar(vals.sort_values("Total"), x="Total", y="Característica", orientation="h")
    fig.update_traces(marker=dict(color=OFERTA_COLOR_BARRES))
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_energetica(df):
    if df.empty:
        return oferta_fig_no_disponible("Qualificació energètica", "No hi ha dades.")
    taula = df["QENERGC"].value_counts().reset_index()
    taula.columns = ["Grup", "Habitatges"]
    fig = px.pie(taula, values="Habitatges", names="Grup", hole=0.4, color_discrete_sequence=PLOTLY_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_calefaccio(df):
    cols = ["De gasoil", "De gas natural", "De propà", "D'electricitat", "No s'indica tipus"]
    taula = df[cols].sum().reset_index()
    taula.columns = ["Tipus", "Total"]
    fig = px.pie(taula, values="Total", names="Tipus", hole=0.4, color_discrete_sequence=PLOTLY_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label", sort=False)
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_aparcament(df):
    taula = df["APAR"].value_counts().reset_index()
    taula.columns = ["Tipus", "Total"]
    fig = px.pie(taula, values="Total", names="Tipus", hole=0.4, color_discrete_sequence=[OFERTA_COLOR_CLAR, OFERTA_COLOR_FOSC])
    fig.update_traces(textposition="outside", textinfo="percent+label", sort=False)
    return oferta_preparar_fig(fig)


@st.cache_data(show_spinner=False)
def oferta_grafic_evolucio(df_final, nivell, geo, variable, any_ini, any_fin):
    df_ev = df_final[(df_final["Nivell"] == nivell) & (df_final["GEO"] == geo) & (df_final["Variable"] == variable) & (df_final["Any"].between(any_ini, any_fin))]
    fig = px.bar(df_ev, x="Any", y="Valor", color="Tipologia", barmode="group", color_discrete_sequence=PLOTLY_PALETTE[:3])
    fig.update_xaxes(type="category")
    return oferta_preparar_fig(fig)


@st.cache_resource(show_spinner="Carregant el mapa...")
def oferta_load_shp(p, tol=8e-4):
    shp = gpd.read_file(p)
    if "codiine" in shp.columns:
        shp["municipi"] = shp["codiine"].astype(int)
    elif "municipi" in shp.columns:
        shp["municipi"] = shp["municipi"].astype(int)
    else:
        return None
    shp["geometry"] = shp.geometry.simplify(tol, preserve_topology=True)
    return shp


def oferta_etiqueta_metrica_mapa(variable, unitats):
    if variable == "Unitats":
        return "Habitatges en oferta"
    if pd.isna(unitats) or str(unitats).strip() == "":
        return "Valor"
    return str(unitats)


@st.cache_data(show_spinner=False)
def oferta_prep_map_df(df_final, any_estudi, tipologia, variable):
    df = df_final[(df_final["Any"] == int(any_estudi)) & (df_final["Tipologia"] == tipologia) & (df_final["Variable"] == variable) & (df_final["Nivell"] == "Municipis")][["codiine", "GEO", "Valor", "Unitats"]].copy()
    df.columns = ["municipi", "nom_muni", "valor", "unitats"]
    df = df.dropna(subset=["municipi"])
    df["municipi"] = df["municipi"].astype(int)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["valor_txt"] = df["valor"].map(lambda x: oferta_format_catala(x) if pd.notna(x) else "")
    df["metrica"] = df.apply(lambda r: oferta_etiqueta_metrica_mapa(variable, r["unitats"]), axis=1)
    return df


@st.cache_resource(show_spinner=False)
def oferta_build_tmp(_shp, df_map):
    shp = _shp.copy()
    if "nom_muni" in shp.columns:
        shp = shp.rename(columns={"nom_muni": "nom_muni_shp"})
    tmp = shp.merge(df_map, on="municipi", how="left")
    if "nom_muni_shp" in tmp.columns:
        tmp["nom_muni"] = tmp["nom_muni"].fillna(tmp["nom_muni_shp"])
    tmp["valor_txt"] = tmp.get("valor_txt", pd.Series(index=tmp.index, dtype="string")).fillna("")
    tmp["metrica"] = tmp.get("metrica", pd.Series(index=tmp.index, dtype="string")).fillna("")
    return tmp


def oferta_color_mapa(valor, minim, maxim):
    if valor is None or pd.isna(valor):
        return OFERTA_COLOR_SURFACE
    if maxim == minim:
        return OFERTA_COLOR_BARRES
    ratio = (float(valor) - minim) / (maxim - minim)
    rgba = colors.LinearSegmentedColormap.from_list("oferta_paleta_mapa", [OFERTA_COLOR_CLAR, OFERTA_COLOR_FOSC])(ratio)
    return colors.to_hex(rgba)


def oferta_folium_map(tmp, title, h=760):
    tiles = "CartoDB dark_matter" if st.session_state.get("theme") == "dark" else "CartoDB voyager"
    m = folium.Map([41.7, 1.6], zoom_start=8, tiles=tiles, width="100%", height=f"{h}px")
    vals = tmp["valor"].dropna()
    minim = float(vals.min()) if not vals.empty else 0.0
    maxim = float(vals.max()) if not vals.empty else 0.0
    folium.GeoJson(
        tmp,
        name=title,
        tooltip=folium.GeoJsonTooltip(fields=["nom_muni", "valor_txt", "metrica"], aliases=["Municipi:", "Valor:", "Mètrica:"], localize=True, sticky=True),
        style_function=lambda feature: {
            "fillColor": oferta_color_mapa(feature["properties"].get("valor"), minim, maxim),
            "color": OFERTA_COLOR_MUTED,
            "weight": 0.35,
            "fillOpacity": 0.78 if feature["properties"].get("valor") is not None else 0.35,
        },
        highlight_function=lambda feature: {"weight": 1.4, "color": OFERTA_COLOR_FOSC, "fillOpacity": 0.9},
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


@st.cache_data(show_spinner=False)
def oferta_preparar_punts_habitatges(dades_totals, any_estudi, tipologia):
    cols = ["latitude", "longitude", "Municipi", "Preu mitjà", "Preu m2 útil", "Superfície útil"]
    df = dades_totals[
        (dades_totals["Any"] == int(any_estudi))
        & (dades_totals["TIPOG"] == tipologia)
        & dades_totals["latitude"].notna()
        & dades_totals["longitude"].notna()
    ][cols].copy()
    return df.to_dict("records")


def oferta_popup_habitatge(row):
    return f"""
    <div style='font-family: Arial, sans-serif; min-width: 190px;'>
      <b>{row.get('Municipi', 'Habitatge')}</b><br>
      Preu mitjà: {oferta_format_num(row.get('Preu mitjà'), 0)} €<br>
      Preu m² útil: {oferta_format_num(row.get('Preu m2 útil'), 0)} €/m²<br>
      Superfície útil: {oferta_format_num(row.get('Superfície útil'), 1)} m²
    </div>
    """


def oferta_mapa_punts_habitatges(punts, h=680):
    tiles = "CartoDB dark_matter" if st.session_state.get("theme") == "dark" else "CartoDB positron"
    m = folium.Map([41.7, 1.6], zoom_start=8, tiles=tiles, width="100%", height=f"{h}px")
    dades_fast = [
        [row["latitude"], row["longitude"], oferta_popup_habitatge(row).replace("\n", " "), f"{row.get('Municipi', '')} · {oferta_format_num(row.get('Preu m2 útil'), 0)} €/m²"]
        for row in punts
    ]
    callback = """
    function (row) {
        var marker = L.marker(new L.LatLng(row[0], row[1]));
        marker.bindPopup(row[2]);
        marker.bindTooltip(row[3]);
        return marker;
    }
    """
    FastMarkerCluster(dades_fast, callback=callback, name="Habitatges en oferta").add_to(m)
    return m


if selected == "Estudi d'oferta d'obra nova APCE":
    oferta_dades_totals = oferta_carregant_dades()
    oferta_dades_2025 = oferta_crear_bases_any(oferta_dades_totals, 2025)
    oferta_dades_2026 = oferta_crear_bases_any(oferta_dades_totals, 2026)
    oferta_df_final = oferta_construir_df_final(oferta_dades_totals)

    st.subheader("ESTUDI D'OFERTA D'OBRA NOVA APCE")
    st.markdown('<div class="oferta-menu-anchor"></div>', unsafe_allow_html=True)
    oferta_selected = st.radio("Secció", OFERTA_PAGINES, horizontal=True, label_visibility="collapsed", key="oferta_menu")

    if oferta_selected == "Catalunya":
        left, right = st.columns((1, 1))
        with left:
            selected_edition = st.radio("**Any**", OFERTA_EDICIONS, index=OFERTA_EDICIONS.index("2026"), horizontal=True, key="oferta_any_catalunya")
        with right:
            st.markdown("**Apartats**")
            st.markdown(
                '<div class="viab-toc">'
                '<a href="#oferta-cat-introduccio">Introducció</a>'
                '<a href="#oferta-cat-caracteristiques">Característiques</a>'
                '<a href="#oferta-cat-qualitats">Qualitats i equipaments</a>'
                '<a href="#oferta-cat-preus">Superfície i preus</a>'
                '<a href="#oferta-cat-comparativa">Comparativa 2025–2026</a>'
                '</div>',
                unsafe_allow_html=True,
            )

        dades = oferta_dades_any(oferta_dades_2025, oferta_dades_2026, selected_edition)

        st.markdown('<div id="oferta-cat-introduccio" class="viab-anchor"></div>', unsafe_allow_html=True)
        oferta_titol_seccio("Introducció")
        st.write(oferta_text_resum_cat(dades, selected_edition), unsafe_allow_html=True)
        oferta_mostra_text_informe("introduccio", selected_edition)

        _oferta_shp = oferta_load_shp(SHAPEFILE_MUN)
        mapa_left, mapa_right = st.columns((1, 1))
        with mapa_left:
            oferta_mostra("Mapa provincial de l'oferta d'habitatges", oferta_mapa_provincial_oferta(dades, _oferta_shp))
        with mapa_right:
            oferta_mostra("Mapa municipal de l'oferta d'habitatges", oferta_mapa_municipal_oferta(dades, _oferta_shp))

        left_col, right_col = st.columns((1, 1))
        with left_col:
            oferta_mostra("Nombre d'habitatges en oferta per província a Catalunya", oferta_grafic_provincies(dades))
        with right_col:
            oferta_mostra("Nombre d'habitatges en oferta per municipis a Catalunya", oferta_grafic_top_municipis(dades))

        st.markdown('<div id="oferta-cat-caracteristiques" class="viab-anchor"></div>', unsafe_allow_html=True)
        oferta_titol_seccio("Característiques")
        st.write("Principals tipologies dels habitatges en oferta segons el nombre de dormitoris i banys.")
        oferta_mostra_text_informe("tipologies", selected_edition)
        st_plotly_chart(oferta_grafic_principals_tipologies(dades), use_container_width=True, responsive=True)

        st.markdown('<div id="oferta-cat-qualitats" class="viab-anchor"></div>', unsafe_allow_html=True)
        oferta_titol_seccio("Qualitats i equipaments")
        oferta_mostra("Qualitats i equipaments dels habitatges en oferta", oferta_grafic_qualitats_equipaments(dades))

        st.markdown('<div id="oferta-cat-preus" class="viab-anchor"></div>', unsafe_allow_html=True)
        oferta_titol_seccio("Superfície i preus")
        r = oferta_resum_basic(dades)
        st.write(f"""
        <p>
        La mitjana de la superfície útil dels habitatges en venda és de {oferta_format_num(r['superficie'],1)} m², amb un preu mitjà de {oferta_format_num(r['preu'],0)} € i un preu mitjà de {oferta_format_num(r['preum2'],0)} €/m² útil.
        Els gràfics mostren aquests indicadors per tipologia i nombre de dormitoris.
        </p>
        """, unsafe_allow_html=True)
        left_col, right_col = st.columns((1, 1))
        with left_col:
            oferta_mostra("Preu mitjà per tipologia d'habitatge (€)", oferta_grafic_mitjanes(dades, "Preu mitjà", "Preu mitjà"))
        with right_col:
            oferta_mostra("Preu per m² útil per tipologia d'habitatge (€/m² útil)", oferta_grafic_mitjanes(dades, "Preu m2 útil", "Preu per m² útil"))
        oferta_mostra("Superfície útil per tipologia d'habitatge (m² útil)", oferta_grafic_mitjanes(dades, "Superfície útil", "Superfície útil"))

        st.markdown('<div id="oferta-cat-comparativa" class="viab-anchor"></div>', unsafe_allow_html=True)
        oferta_titol_seccio("Comparativa 2025–2026")
        if selected_edition == "2025":
            st.warning("Les dades només estan disponibles des de 2025 en aquesta app. No es pot calcular la comparativa amb 2024.")
            st.markdown(taula_html_es(oferta_taula_comparativa(oferta_df_final, "Catalunya", None, 2025, 2025), precision=0), unsafe_allow_html=True)
        else:
            st.write("<p>La comparativa es calcula a partir de les dades deduplicades dels dos semestres analitzats (2025 i 2026). La lectura s'ha de fer com una comparació entre semestres equivalents.</p>", unsafe_allow_html=True)
            oferta_mostra_text_informe("territori", selected_edition)
            oferta_mostra("Variació anual dels principals indicadors per tipologia d'habitatge (%)", oferta_grafic_variacio_anual(oferta_dades_2026, oferta_dades_2025))
            taula_cat = oferta_taula_comparativa(oferta_df_final, "Catalunya", None, 2025, 2026)
            st.markdown(taula_html_es(taula_cat, precision=0), unsafe_allow_html=True)
            st.markdown(oferta_filedownload(taula_cat, "Estudi_oferta_Catalunya_APCE_2025_2026.xlsx"), unsafe_allow_html=True)

    if oferta_selected == "Províncies i àmbits":
        left, center, right = st.columns((1, 1, 1))
        with left:
            selected_edition = st.radio("**Any**", OFERTA_EDICIONS, index=OFERTA_EDICIONS.index("2026"), horizontal=True, key="oferta_any_prov")
        with center:
            selected_option = st.radio("**Àrea geogràfica**", ["Províncies", "Àmbits territorials"], horizontal=True, key="oferta_geo_opcio")
        dades = oferta_dades_any(oferta_dades_2025, oferta_dades_2026, selected_edition)
        with right:
            if selected_option == "Províncies":
                prov_names = sorted(dades["PROVINCIA"].dropna().unique().tolist())
                selected_geo = st.selectbox("**Selecciona una província**", prov_names, index=prov_names.index("Barcelona") if "Barcelona" in prov_names else 0, key="oferta_prov_sel")
                columna_geo = "PROVINCIA"
            else:
                ambit_names = sorted(dades["TERRITORI"].dropna().unique().tolist())
                selected_geo = st.selectbox("**Selecciona un àmbit territorial**", ambit_names, index=ambit_names.index("Metropolità") if "Metropolità" in ambit_names else 0, key="oferta_ambit_sel")
                columna_geo = "TERRITORI"

        if selected_option == "Províncies":
            oferta_titol_seccio(f"Província de {selected_geo}")
            oferta_mostra_text_informe("provincies", selected_edition)
        else:
            oferta_titol_seccio(OFERTA_TITOL_AMBIT.get(selected_geo, f"Àmbit de {selected_geo}"))
            oferta_mostra_text_informe("ambits", selected_edition)
        st.markdown(oferta_text_resum_geo(dades, selected_geo, columna_geo, selected_edition))
        nivell_geo = "Àmbits territorials" if selected_option == "Àmbits territorials" else "Províncies"
        taula_geo = oferta_taula_comparativa(oferta_df_final, nivell_geo, selected_geo, 2025, int(selected_edition))
        st.markdown(taula_html_es(taula_geo, precision=0), unsafe_allow_html=True)
        st.markdown(oferta_filedownload(taula_geo, f"Estudi_oferta_APCE_{selected_geo}.xlsx"), unsafe_allow_html=True)

        df_geo = oferta_filtra_geo(dades, columna_geo, selected_geo)
        fila_1_left, fila_1_right = st.columns((1, 1))
        with fila_1_left:
            oferta_mostra("Proporció d'habitatges segons tipologia", oferta_grafic_tipologia_donut(df_geo))
        with fila_1_right:
            oferta_mostra("Habitatges d'obra nova / rehabilitació integral", oferta_grafic_obra_nova(df_geo))

        fila_2_left, fila_2_right = st.columns((1, 1))
        with fila_2_left:
            oferta_mostra("Habitatges a la venda segons número d'habitacions", oferta_grafic_dormitoris(df_geo))
        with fila_2_right:
            oferta_mostra("Qualitats i equipaments dels habitatges en oferta", oferta_grafic_qualitats_equipaments(df_geo))

    if oferta_selected == "Municipis":
        left, center = st.columns((0.8, 1.8))
        with left:
            selected_edition = st.radio("**Any**", OFERTA_EDICIONS, index=OFERTA_EDICIONS.index("2026"), horizontal=True, key="oferta_any_mun")
        dades = oferta_dades_any(oferta_dades_2025, oferta_dades_2026, selected_edition)
        with center:
            mun_names = sorted(dades["Municipi"].dropna().unique().tolist())
            selected_mun_oferta = st.selectbox("**Selecciona un municipi**", mun_names, index=mun_names.index("Barcelona") if "Barcelona" in mun_names else 0, key="oferta_mun_sel")

        oferta_titol_seccio(f"Municipi de {selected_mun_oferta}")
        st.write(oferta_text_resum_mun_dis(dades, selected_mun_oferta, "Municipi", selected_edition), unsafe_allow_html=True)

        df_geo = oferta_filtra_geo(dades, "Municipi", selected_mun_oferta)
        hist_left, hist_right = st.columns((1, 1))
        with hist_left:
            oferta_mostra("Distribució de Preus per m² útil", oferta_grafic_distribucio(df_geo, "Preu m2 útil"))
        with hist_right:
            oferta_mostra("Distribució de Superfície útil", oferta_grafic_distribucio(df_geo, "Superfície útil"))
        st.caption(OFERTA_NOTA_MEDIANA)

        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.markdown("**Preus per m² útil segons nombre d'habitacions i lavabos**")
            st.markdown(oferta_matriu_hab_lav(df_geo, "Preu m2 útil").to_html(), unsafe_allow_html=True)
            oferta_mostra("Característiques principals dels habitatges en oferta", oferta_grafic_caracteristiques(df_geo))
            oferta_mostra("Qualitats i equipaments dels habitatges en oferta", oferta_grafic_qualitats_equipaments(df_geo))
            oferta_mostra("Habitatges a la venda segons número d'habitacions", oferta_grafic_dormitoris(df_geo))
        with right_col:
            st.markdown("**Superfície en m² útils segons nombre d'habitacions i lavabos**")
            st.markdown(oferta_matriu_hab_lav(df_geo, "Superfície útil").to_html(), unsafe_allow_html=True)
            oferta_mostra("Proporció d'habitatges en oferta a les promocions segons tipologia (%)", oferta_grafic_tipologia_pie(df_geo))
            oferta_mostra("Habitatges a la venda segons número de lavabos", oferta_grafic_lavabos(df_geo))
            oferta_mostra("Plaça d'aparcament inclosa o no en els habitatges en oferta (%)", oferta_grafic_aparcament(df_geo))

        energia_left, energia_right = st.columns((1, 1))
        with energia_left:
            oferta_mostra("Qualificació energètica dels habitatges en oferta (% d'habitatges)", oferta_grafic_energetica(df_geo))
        with energia_right:
            oferta_mostra("Proporció d'habitatges segons el tipus d'instal·lació de calefacció (%)", oferta_grafic_calefaccio(df_geo))

        st.subheader(f"Evolució 2025–2026 · {selected_mun_oferta}")
        taula_mun = oferta_taula_comparativa(oferta_df_final, "Municipis", selected_mun_oferta, 2025, int(selected_edition))
        st.markdown(taula_html_es(taula_mun, precision=0), unsafe_allow_html=True)
        st.markdown(oferta_filedownload(taula_mun, f"Estudi_oferta_APCE_{selected_mun_oferta}.xlsx"), unsafe_allow_html=True)
        ed = int(selected_edition)
        left_col, right_col = st.columns((1, 1))
        with left_col:
            oferta_mostra("Evolució dels habitatges de nova construcció per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Municipis", selected_mun_oferta, "Unitats", 2025, ed))
        with right_col:
            oferta_mostra("Evolució de la superfície útil mitjana per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Municipis", selected_mun_oferta, "Superfície mitjana (m² útils)", 2025, ed))
        left_col, right_col = st.columns((1, 1))
        with left_col:
            oferta_mostra("Evolució del preu de venda per m² útil per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Municipis", selected_mun_oferta, "Preu de venda per m² útil (€)", 2025, ed))
        with right_col:
            oferta_mostra("Evolució del preu venda mitjà per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Municipis", selected_mun_oferta, "Preu mitjà de venda de l'habitatge (€)", 2025, ed))

    if oferta_selected == "Districtes de Barcelona":
        left, right = st.columns((1, 1))
        with left:
            selected_edition = st.radio("**Any**", OFERTA_EDICIONS, index=OFERTA_EDICIONS.index("2026"), horizontal=True, key="oferta_any_dis")
        dades = oferta_dades_any(oferta_dades_2025, oferta_dades_2026, selected_edition)
        bcn = dades[(dades["Municipi"] == "Barcelona") & dades["Nom DIST"].notna()].copy()
        with right:
            dis_names = sorted(bcn["Nom DIST"].dropna().unique().tolist())
            selected_dis = st.selectbox("**Selecciona un districte**", dis_names, index=0 if dis_names else None, key="oferta_dis_sel")

        if not dis_names:
            oferta_titol_seccio("Districtes de Barcelona")
            st.warning("No hi ha districtes de Barcelona disponibles a la base de dades.")
        else:
            oferta_titol_seccio(f"Districte de {selected_dis}")
            oferta_mostra_text_informe("districtes_barcelona", selected_edition)
            st.write(oferta_text_resum_mun_dis(dades[(dades["Municipi"] == "Barcelona")], selected_dis, "Nom DIST", selected_edition), unsafe_allow_html=True)

            df_geo = oferta_filtra_geo(dades, "Nom DIST", selected_dis)
            hist_left, hist_right = st.columns((1, 1))
            with hist_left:
                oferta_mostra("Distribució de Preus per m² útil", oferta_grafic_distribucio(df_geo, "Preu m2 útil"))
            with hist_right:
                oferta_mostra("Distribució de Superfície útil", oferta_grafic_distribucio(df_geo, "Superfície útil"))
            st.caption(OFERTA_NOTA_MEDIANA)

            left_col, right_col = st.columns((1, 1))
            with left_col:
                st.markdown("**Preus per m² útil segons nombre d'habitacions i lavabos**")
                st.markdown(oferta_matriu_hab_lav(df_geo, "Preu m2 útil").to_html(), unsafe_allow_html=True)
                oferta_mostra("Característiques principals dels habitatges en oferta", oferta_grafic_caracteristiques(df_geo))
                oferta_mostra("Qualitats i equipaments dels habitatges en oferta", oferta_grafic_qualitats_equipaments(df_geo))
                oferta_mostra("Habitatges a la venda segons número d'habitacions", oferta_grafic_dormitoris(df_geo))
            with right_col:
                st.markdown("**Superfície en m² útils segons nombre d'habitacions i lavabos**")
                st.markdown(oferta_matriu_hab_lav(df_geo, "Superfície útil").to_html(), unsafe_allow_html=True)
                oferta_mostra("Proporció d'habitatges en oferta a les promocions segons tipologia (%)", oferta_grafic_tipologia_pie(df_geo))
                oferta_mostra("Habitatges a la venda segons número de lavabos", oferta_grafic_lavabos(df_geo))
                oferta_mostra("Plaça d'aparcament inclosa o no en els habitatges en oferta (%)", oferta_grafic_aparcament(df_geo))

            energia_left, energia_right = st.columns((1, 1))
            with energia_left:
                oferta_mostra("Qualificació energètica dels habitatges en oferta (% d'habitatges)", oferta_grafic_energetica(df_geo))
            with energia_right:
                oferta_mostra("Proporció d'habitatges segons el tipus d'instal·lació de calefacció (%)", oferta_grafic_calefaccio(df_geo))

            st.subheader(f"Evolució 2025–2026 · {selected_dis}")
            taula_dis = oferta_taula_comparativa(oferta_df_final, "Districtes de Barcelona", selected_dis, 2025, int(selected_edition))
            st.markdown(taula_html_es(taula_dis, precision=0), unsafe_allow_html=True)
            st.markdown(oferta_filedownload(taula_dis, f"Estudi_oferta_APCE_{selected_dis}.xlsx"), unsafe_allow_html=True)
            ed = int(selected_edition)
            left_col, right_col = st.columns((1, 1))
            with left_col:
                oferta_mostra("Evolució dels habitatges de nova construcció per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Districtes de Barcelona", selected_dis, "Unitats", 2025, ed))
            with right_col:
                oferta_mostra("Evolució de la superfície útil mitjana per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Districtes de Barcelona", selected_dis, "Superfície mitjana (m² útils)", 2025, ed))
            left_col, right_col = st.columns((1, 1))
            with left_col:
                oferta_mostra("Evolució del preu de venda per m² útil per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Districtes de Barcelona", selected_dis, "Preu de venda per m² útil (€)", 2025, ed))
            with right_col:
                oferta_mostra("Evolució del preu venda mitjà per tipologia d'habitatge", oferta_grafic_evolucio(oferta_df_final, "Districtes de Barcelona", selected_dis, "Preu mitjà de venda de l'habitatge (€)", 2025, ed))

    if oferta_selected == "Mapa interactiu":
        opc = {
            "Habitatges en oferta": "Unitats",
            "Superfície mitjana": "Superfície mitjana (m² útils)",
            "Preu mitjà": "Preu mitjà de venda de l'habitatge (€)",
            "Preu m² útil": "Preu de venda per m² útil (€)",
        }
        tipus_mapa = st.radio("Tipus de mapa", ["Mapa de municipis", "Mapa d'habitatges en oferta"], horizontal=True, label_visibility="collapsed", key="oferta_tipus_mapa")

        if tipus_mapa == "Mapa de municipis":
            with st.container(border=True):
                left, mid, right = st.columns(3)
                with left:
                    label = st.selectbox("Indicador", list(opc.keys()), key="oferta_mapa_indicador")
                with mid:
                    any_mapa = st.selectbox("Any", [2025, 2026], index=1, key="oferta_mapa_any")
                with right:
                    tipologia = st.selectbox("Tipologia", sorted(oferta_df_final["Tipologia"].dropna().str.lower().str.capitalize().unique().tolist()), key="oferta_mapa_tipologia")

            tipologia_upper = tipologia.upper()
            variable_mapa = opc[label]
            df_map = oferta_prep_map_df(oferta_df_final, any_mapa, tipologia_upper, variable_mapa)

            oferta_titol_seccio("Mapa de municipis")
            _oferta_shp_mapa = oferta_load_shp(SHAPEFILE_MUN)
            if _oferta_shp_mapa is not None:
                tmp = oferta_build_tmp(_oferta_shp_mapa, df_map)
                m = oferta_folium_map(tmp, f"{label} · {tipologia.lower().capitalize()} · {any_mapa}", h=760)
                st_folium(m, use_container_width=True, height=760, returned_objects=[])
            else:
                st.warning("El shapefile no conté un camp municipal compatible.")
        else:
            with st.container(border=True):
                left, right = st.columns(2)
                with left:
                    any_punts = st.selectbox("Any", [2025, 2026], index=1, key="oferta_any_punts")
                with right:
                    tipologies_punts = sorted(oferta_dades_totals["TIPOG"].dropna().unique().tolist())
                    index_tipologia = tipologies_punts.index("Habitatges plurifamiliars") if "Habitatges plurifamiliars" in tipologies_punts else 0
                    tipologia_punts = st.selectbox("Tipologia", tipologies_punts, index=index_tipologia, key="oferta_tipologia_punts")

            oferta_titol_seccio("Mapa d'habitatges en oferta")
            punts = oferta_preparar_punts_habitatges(oferta_dades_totals, any_punts, tipologia_punts)
            if punts:
                st_folium(oferta_mapa_punts_habitatges(punts, h=680), use_container_width=True, height=680, returned_objects=[])
            else:
                st.info("No hi ha coordenades disponibles.")

############################################################  BOTÓ "TORNAR A DALT" (sempre visible) ################################################
# Enllaç fix a baix a la dreta que porta a l'àncora #dalt del principi (desplaçament suau).
# Fora de tots els blocs "if selected==...": es renderitza sempre, independentment de la pestanya activa.
st.markdown('<a href="#dalt" class="boto-amunt" title="Tornar a dalt" aria-label="Tornar a dalt">↑</a>', unsafe_allow_html=True)
