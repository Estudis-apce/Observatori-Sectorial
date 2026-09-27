"""Fitxa municipal de "demanda potencial d'habitatge".

Rèplica de la metodologia de l'informe APCE/Diputació de Barcelona (2014): per
cada municipi amb dada real de compravendes d'obra nova (339 dels 947 de
Catalunya), un índex compost (4 subíndexs — Demogràfic/Mobilitat/Socioeconòmic/
Habitatge — + agregat, amb semàfor verd/groc/vermell) i ~53 indicadors amb
comparativa municipi/comarca/província.

MÒDUL INDEPENDENT I AUTOCONTINGUT: no importa res d'APP_Dades.py ni de
Streamlit/reportlab — només `pandas`/`numpy`/`json`/`pathlib`. Si es vol treure
aquesta funcionalitat de l'app, n'hi ha prou d'esborrar aquest fitxer (o els 3
JSON que llegeix): els 2 punts d'enganxament a APP_Dades.py estan protegits amb
try/except i es desactiven sols, sense trencar res.

Font de les dades — NOMÉS fitxers ja existents, cap fitxer nou es genera ni es
desa a disc:
  - 3 JSON a Resources/JSON/, còpia directa (sense tocar res) des de
    "Z:\\ESTUDIS\\Informe d'indicadors de demanda potencial d'habitatge\\"
    (carpeta de l'informe, mai modificada des d'aquí):
      indicadores_informe_demanda.json          (947 municipis, indicadors crus
        + subíndexs + semàfors agregats; només 339 tenen
        "transaccions_obra_nova" real — vegeu `disponible()`)
      indicadores_informe_demanda_comarca.json  (39 comarques, agregats)
      indicadores_informe_demanda_provincia.json (4 províncies, agregats)
  - "Z:\\ESTUDIS\\Base de Dades\\Indicadors municipis\\Maestro_postal.csv"
    (llegit directament d'on ja és, sense copiar-lo): dona el mapeig
    municipi -> comarca/província, que el JSON principal no porta perquè
    Genera_fitxes_html.ipynb només el fa servir en memòria i no l'exporta.

Dues coses que Genera_fitxes_html.ipynb calcula "de pas" i mai desa enlloc
(el mapeig geogràfic i el semàfor per indicador individual) es recalculen aquí
en memòria, cada vegada que s'importa el mòdul (`_carregar()`), a partir
d'aquests mateixos fitxers ja existents -- amb la mateixa fórmula exacta del
notebook, sense cap fitxer derivat nou. Si el notebook d'origen es torna a
executar, cal tornar a copiar manualment els 3 JSON de Resources/JSON/; no cal
tocar res més.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
_JSON_DIR = _BASE_DIR / "Resources" / "JSON"

_FITXER_MUN = _JSON_DIR / "indicadores_informe_demanda.json"
_FITXER_COMARCA = _JSON_DIR / "indicadores_informe_demanda_comarca.json"
_FITXER_PROVINCIA = _JSON_DIR / "indicadores_informe_demanda_provincia.json"

# Fitxer extern (no copiat, es llegeix d'on ja és) -- mateix que fa servir
# Genera_fitxes_html.ipynb per obtenir comarca/província de cada municipi.
_FITXER_MAESTRO_POSTAL = Path("Z:/ESTUDIS/Base de Dades/Indicadors municipis/Maestro_postal.csv")
_CODI_PROVINCIA = {"08": "Barcelona", "17": "Girona", "25": "Lleida", "43": "Tarragona"}

# Pesos dels ~26 indicadors individuals amb semàfor -- idèntics a
# Genera_fitxes_html.ipynb (cel·la 4), per recalcular el semàfor per indicador
# en memòria (el JSON principal només porta els 5 semàfors de subíndex/agregat).
_PESOS_DEMOGRAFIC = {
    "variacio_poblacio_pct": 1, "pes_25_35_pct": 0.5, "pes_35_44_pct": 0.5,
    "naixements_pob_pct": 0.5, "nupcialitat_pob_pct": 0.5,
}
_PESOS_MOBILITAT = {
    "immigracio_interior_pob_pct": 1, "immigracio_exterior_pob_pct": 0.5, "emigracio_interior_pob_pct": -1,
    "immigracio_25_35_pct": 0.5, "emigracio_25_35_pct": -0.5,
    "pes_25_35_immigracio_pct": 0.5, "pes_25_35_emigracio_pct": -0.5,
    "immigracio_35_44_pct": 1, "emigracio_35_44_pct": -1,
    "pes_35_44_immigracio_pct": 1, "pes_35_44_emigracio_pct": -1,
}
_PESOS_SOCIOECONOMIC = {
    "atur_pob_pct": -1, "renda_neta_llar": 1, "esforc_compra_pct": -1, "esforc_lloguer_pct": 0.5,
    "diferencial_esforc_compra_lloguer": -0.5,
    "estrangers_extracomunitaris_pob_pct": -0.5, "estrangers_ue_russia_pob_pct": 0.5,
}
_PESOS_HABITATGE = {
    "transaccions_obra_nova_pob_pct": 0.5, "estoc_estimat_obra_nova": -1,
    "habitatges_en_oferta": -1, "diferencial_esforc_compra_lloguer": -0.5,
}
_PESOS_TOTS = {**_PESOS_DEMOGRAFIC, **_PESOS_MOBILITAT, **_PESOS_SOCIOECONOMIC, **_PESOS_HABITATGE}


def _zscore(serie: pd.Series) -> pd.Series:
    return (serie - serie.mean(skipna=True)) / serie.std(skipna=True)


def _semafor_de_z(z) -> Optional[str]:
    if pd.isna(z):
        return None
    if z >= 0.25:
        return "verd"
    if z <= -0.25:
        return "vermell"
    return "groc"

# ========== BLOCS (columna, etiqueta, format) — idèntic a Genera_fitxes_html.ipynb ==========
BLOC_DEMOGRAFIC = [
    ("poblacio", "Població", "num"),
    ("variacio_poblacio_pct", "Variació de la població", "pct"),
    ("pes_25_35_pct", "Població 25-35 anys / població", "pct"),
    ("pes_35_44_pct", "Població 35-44 anys / població", "pct"),
    ("naixements_pob_pct", "Naixements / població", "pct"),
    ("nupcialitat_pob_pct", "Nupcialitat / població", "pct"),
]

BLOC_MOBILITAT = [
    ("immigracio_interior_pob_pct", "Immigració interior / població", "pct"),
    ("immigracio_exterior_pob_pct", "Immigració exterior / població", "pct"),
    ("emigracio_interior_pob_pct", "Emigració interior / població", "pct"),
    ("emigracio_exterior_pob_pct", "Emigració exterior / població", "pct"),
]

BLOC_SOCIOECONOMIC = [
    ("atur_registrat", "Atur registrat", "num"),
    ("atur_pob_pct", "Atur registrat / població", "pct"),
    ("renda_neta_llar", "Renda neta mitjana per llar", "eur"),
    ("renda_neta_persona", "Renda neta mitjana per persona", "eur"),
    ("esforc_compra_pct", "Esforç econòmic accés obra nova", "pct"),
    ("esforc_lloguer_pct", "Esforç econòmic accés lloguer", "pct"),
    ("diferencial_esforc_compra_lloguer", "Diferencial esforç obra nova / lloguer", "pp"),
]

BLOC_HABITATGE = [
    ("preu_obra_nova_milers", "Preu mitjà obra nova", "eurm"),
    ("preu_obra_nova_variacio_pct", "Preu mitjà obra nova (variació interanual)", "pct"),
    ("preu_m2_construit_obra_nova", "Preu m² construït obra nova", "eur"),
    ("preu_m2_util_obra_nova", "Preu m² útil obra nova (oferta activa, Atlas)", "eur"),
    ("preu_lloguer_mensual", "Preu mitjà lloguer mensual", "eur"),
    ("transaccions_obra_nova_pob_pct", "Transaccions habitatge nou / població", "pct"),
    ("transaccions_segona_ma_pob_pct", "Transaccions habitatges usats / població", "pct"),
    ("habitatges_finalitzats", "Habitatges finalitzats", "num"),
    ("contractes_lloguer", "Contractes de lloguer", "num"),
    ("contractes_lloguer_pob_pct", "Contractes de lloguer / població", "pct"),
    ("estoc_estimat_obra_nova", "Estimació estoc obra nova / població", "pct"),
    ("pes_estoc_habitatge_construit_pct", "Pes de l'estoc sobre habitatge construït", "pct"),
    ("habitatges_en_oferta_abs", "Habitatges nous en oferta (recompte)", "num"),
    ("habitatges_en_oferta", "Habitatges nous en oferta / població", "pct"),
    ("preu_maxim_modul_hpo_general", "Preu màxim mòdul HPO (general)", "eur"),
    ("preu_maxim_modul_hpo_concertat", "Preu màxim mòdul HPO (concertat)", "eur"),
    ("diferencia_lliure_protegit_general_pct", "Diferència lliure / protegit (general)", "pct"),
    ("diferencia_lliure_protegit_concertat_pct", "Diferència lliure / protegit (concertat)", "pct"),
]

BLOC_CENS2021 = [
    ("pct_propietat_c21", "Habitatges en propietat", "pct"),
    ("pct_lloguer_c21", "Habitatges en lloguer", "pct"),
    ("superficie_mitjana_m2_c21", "Superfície mitjana", "m2"),
    ("mida_mitjana_llar_c21", "Mida mitjana de la llar", "pers"),
    ("edat_mitjana_habitatge_c21", "Edat mitjana de l'habitatge", "anys"),
    ("pct_estrangera_c21", "Població estrangera", "pct"),
    ("pct_educacio_superior_c21", "Població amb educació superior", "pct"),
    ("pct_no_principals_c21", "Habitatges no principals", "pct"),
]

_TOTS_ELS_BLOCS = [
    ("Demogràfic", BLOC_DEMOGRAFIC),
    ("Mobilitat residencial", BLOC_MOBILITAT),
    ("Socioeconòmic", BLOC_SOCIOECONOMIC),
    ("Habitatge", BLOC_HABITATGE),
    ("Cens 2021 (informatiu, fora de l'índex)", BLOC_CENS2021),
]

# ========== FORMAT I SEMÀFOR ==========
# Paleta harmonitzada amb la identitat "Forest & Coral" de l'app (CSS_COLORS a
# APP_Dades.py: primary #C1571E, accent #E3A94C, brand_dark #2F4A38) en lloc
# dels verds/grocs/vermells genèrics tipus Material Design.
COLORS_SEMAFOR = {
    "verd": ("#2F4A38", "rgba(47, 74, 56, 0.12)"),
    "groc": ("#95610C", "rgba(227, 169, 76, 0.22)"),
    "vermell": ("#9C3B22", "rgba(156, 59, 34, 0.14)"),
}
COLOR_SENSE_DADES = ("#8a8074", "rgba(138, 128, 116, 0.12)")


def formata_valor(valor, format_tipus: str) -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return "Dada no disponible"
    if format_tipus == "pct":
        return f"{valor:.1f}%"
    if format_tipus == "pp":
        return f"{valor:+.1f} p.p."
    if format_tipus == "eur":
        return f"{valor:,.0f} €".replace(",", ".")
    if format_tipus == "eurm":
        return f"{valor * 1000:,.0f} €".replace(",", ".")
    if format_tipus == "num":
        return f"{valor:,.0f}".replace(",", ".")
    if format_tipus == "m2":
        return f"{valor:.0f} m²"
    if format_tipus == "pers":
        return f"{valor:.2f} persones"
    if format_tipus == "anys":
        return f"{valor:.0f} anys"
    return str(valor)


def _es_nan(valor) -> bool:
    return valor is None or (isinstance(valor, float) and pd.isna(valor))


# ========== CÀRREGA (foto fixa, cache simple a nivell de mòdul) ==========
_cache: dict = {}


def _carregar_geo() -> dict:
    """Mapeig municipi -> comarca/província, llegit directament de
    Maestro_postal.csv (mateix fitxer i mateixa transformació que fa
    Genera_fitxes_html.ipynb, cel·la 2) -- no es desa enlloc, es recalcula
    cada vegada que es carrega el mòdul."""
    maestro = pd.read_csv(_FITXER_MAESTRO_POSTAL, encoding="UTF-8", sep=";")
    maestro["Provincia"] = maestro["Codi"].astype(str).str.zfill(6).str[:2].map(_CODI_PROVINCIA)
    maestro = maestro.rename(columns={"Nom": "municipi", "Nom comarca": "comarca"})
    return {
        r["municipi"]: {"comarca": r["comarca"], "provincia": r["Provincia"]}
        for r in maestro[["municipi", "comarca", "Provincia"]].to_dict("records")
    }


def _calcular_semafors_individuals(mun_index: dict) -> None:
    """Semàfor per als ~26 indicadors individuals ponderats, calculat en
    memòria amb la mateixa fórmula de Genera_fitxes_html.ipynb (cel·la 4:
    z-score × signe del pes, sobre els 339 municipis amb dada real) -- el
    JSON principal només porta els 5 semàfors de subíndex/agregat. Modifica
    `mun_index` in-place, afegint claus "semafor_<indicador>"."""
    df = pd.DataFrame(mun_index.values())
    df_339 = df[df["transaccions_obra_nova"].notna()]
    for col, pes in _PESOS_TOTS.items():
        if col not in df_339.columns:
            continue
        signe = 1 if pes >= 0 else -1
        semafors = (_zscore(df_339[col]) * signe).apply(_semafor_de_z)
        for municipi, valor in zip(df_339["municipi"], semafors):
            mun_index[municipi][f"semafor_{col}"] = valor


def _carregar() -> dict:
    """Llegeix els 3 JSON de Resources/JSON/ un sol cop per procés, i hi
    afegeix (en memòria, sense desar res) la comarca/província de
    Maestro_postal.csv i el semàfor per indicador individual. Si algun fitxer
    no existeix, llança l'excepció cap amunt — els punts d'enganxament a
    APP_Dades.py ho capturen amb try/except."""
    if _cache:
        return _cache
    with open(_FITXER_MUN, encoding="utf-8") as f:
        mun_records = json.load(f)
    with open(_FITXER_COMARCA, encoding="utf-8") as f:
        comarca_records = json.load(f)
    with open(_FITXER_PROVINCIA, encoding="utf-8") as f:
        provincia_records = json.load(f)

    mun_index = {r["municipi"]: r for r in mun_records}
    _calcular_semafors_individuals(mun_index)

    _cache["mun"] = mun_index
    _cache["comarca"] = {r["comarca"]: r for r in comarca_records}
    _cache["provincia"] = {r["provincia"]: r for r in provincia_records}
    _cache["geo"] = _carregar_geo()
    _cache["n_universe"] = sum(1 for r in mun_records if not _es_nan(r.get("transaccions_obra_nova")))
    return _cache


def text_metodologia() -> str:
    """Explicació breu de com es construeix l'índex i el semàfor -- sense cap
    referència a la font/informe original (a petició de l'usuari). Text pla,
    reutilitzat tant a `render_html()` (pantalla) com al PDF."""
    dades = _carregar()
    n = dades.get("n_universe", "")
    return (
        f"Cada subíndex combina diversos indicadors estandarditzats (z-score) respecte "
        f"als {n} municipis amb dada real de compravendes d'obra nova; l'índex agregat "
        f"combina els 4 subíndexs. El semàfor de cada indicador mostra si el municipi "
        f"queda per sobre (verd), a prop (groc) o per sota (vermell) de la mitjana "
        f"d'aquest grup de municipis."
    )


def disponible(municipi: str) -> bool:
    """True si `municipi` té ficha (mateix criteri que Genera_fitxes_html.ipynb:
    dada real de compravendes d'obra nova, no NaN)."""
    try:
        dades = _carregar()
    except Exception:
        return False
    fila = dades["mun"].get(municipi)
    if fila is None:
        return False
    return not _es_nan(fila.get("transaccions_obra_nova"))


def _fila_i_comparatives(municipi: str):
    dades = _carregar()
    fila = dades["mun"].get(municipi)
    if fila is None:
        return None, None, None, None, None
    geo = dades["geo"].get(municipi, {})
    comarca = geo.get("comarca")
    provincia = geo.get("provincia")
    fila_comarca = dades["comarca"].get(comarca)
    fila_provincia = dades["provincia"].get(provincia)
    return fila, comarca, provincia, fila_comarca, fila_provincia


def resum_indexs(municipi: str) -> list[dict]:
    """Els 5 índexs (4 subíndexs + agregat) amb nom/valor/semàfor."""
    fila, _, _, _, _ = _fila_i_comparatives(municipi)
    if fila is None:
        return []
    return [
        {"nom": "Demogràfic", "valor": fila.get("subindex_demografic"), "semafor": fila.get("semafor_demografic"), "agregat": False},
        {"nom": "Mobilitat", "valor": fila.get("subindex_mobilitat"), "semafor": fila.get("semafor_mobilitat"), "agregat": False},
        {"nom": "Socioeconòmic", "valor": fila.get("subindex_socioeconomic"), "semafor": fila.get("semafor_socioeconomic"), "agregat": False},
        {"nom": "Habitatge", "valor": fila.get("subindex_habitatge"), "semafor": fila.get("semafor_habitatge"), "agregat": False},
        {"nom": "Índex agregat", "valor": fila.get("index_agregat"), "semafor": fila.get("semafor_agregat"), "agregat": True},
    ]


def blocs_dataframes(municipi: str) -> list[tuple[str, pd.DataFrame]]:
    """Un DataFrame per bloc (índex = indicador, columnes = municipi/comarca/
    província ja formatejats com a text). Pensat per `st.dataframe()` i per
    convertir-se directament en items ("table", (titol, df)) del PDF.

    Nota: es construeix amb llistes posicionals (no un dict per fila) perquè el
    nom del municipi pot coincidir amb el de la seva pròpia província (Barcelona,
    Girona, Lleida, Tarragona) — amb un dict, la clau repetida es fusionaria i es
    perdria la columna."""
    fila, comarca, provincia, fila_comarca, fila_provincia = _fila_i_comparatives(municipi)
    if fila is None:
        return []
    nom_comarca = comarca or "Comarca"
    nom_provincia = provincia or "Província"
    columnes = ["Indicador", fila["municipi"], nom_comarca, nom_provincia]
    resultat = []
    for titol, variables in _TOTS_ELS_BLOCS:
        files = []
        for col, etiqueta, format_tipus in variables:
            files.append([
                etiqueta,
                formata_valor(fila.get(col), format_tipus),
                formata_valor((fila_comarca or {}).get(col), format_tipus),
                formata_valor((fila_provincia or {}).get(col), format_tipus),
            ])
        df = pd.DataFrame(files, columns=columnes).set_index("Indicador")
        resultat.append((titol, df))
    return resultat


# ========== RENDER HTML (per a pantalla) ==========
# Sense targeta blanca ni ombra: el contenidor és transparent i s'integra a la
# pàgina (fons/color de text hereten les variables --app-* que ja injecta
# apply_theme_css() a APP_Dades.py, així que segueix el tema clar/fosc de
# l'app). Les taules NO porten estil propi -- reutilitzen la regla `table` ja
# definida a main.css (capçalera var(--app-primary), files alternades), per
# això s'escriuen amb <thead>/<tbody> igual que la resta de taules de l'app.
_CSS_FITXA = """
.fitxa-dp-contenidor { color: var(--app-text); }
.fitxa-dp-contenidor h1 { margin: 0 0 4px 0; font-size: 26px; color: var(--app-primary); }
.fitxa-dp-contenidor .fitxa-dp-subtitol { opacity: 0.7; margin-bottom: 20px; font-size: 14px; }
.fitxa-dp-contenidor .fitxa-dp-resum { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 28px; }
.fitxa-dp-contenidor .fitxa-dp-caixa-index { flex: 1; min-width: 140px; border-radius: 8px; padding: 14px 16px; text-align: center; }
.fitxa-dp-contenidor .fitxa-dp-caixa-index .nom { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; opacity: 0.85; }
.fitxa-dp-contenidor .fitxa-dp-caixa-index .valor { font-size: 21px; font-weight: 700; }
.fitxa-dp-contenidor .fitxa-dp-caixa-index.agregat .valor { font-size: 26px; }
.fitxa-dp-contenidor .custom-box { margin: 28px 0 4px 0; }
.fitxa-dp-contenidor table { width: 100%; margin: 8px 0 0 0; }
.fitxa-dp-contenidor th { text-align: left; }
.fitxa-dp-contenidor th.valor, .fitxa-dp-contenidor td.valor { text-align: right; }
.fitxa-dp-contenidor td.sense-dada { opacity: 0.55; font-style: italic; }
.fitxa-dp-contenidor .cercle-semafor { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.fitxa-dp-contenidor .fitxa-dp-metodologia { margin-top: 24px; font-size: 12px; opacity: 0.65; line-height: 1.4; }
"""


def _cercle_semafor_html(valor_semafor: Optional[str]) -> str:
    if not valor_semafor:
        return ""
    color_text, _ = COLORS_SEMAFOR.get(valor_semafor, COLOR_SENSE_DADES)
    return f'<span class="cercle-semafor" style="background:{color_text};"></span>'


def _caixa_index_html(index: dict) -> str:
    color_text, color_fons = COLORS_SEMAFOR.get(index["semafor"], COLOR_SENSE_DADES)
    valor = index["valor"]
    valor_txt = f"{valor:.2f}" if not _es_nan(valor) else "—"
    classe = "fitxa-dp-caixa-index agregat" if index["agregat"] else "fitxa-dp-caixa-index"
    return (
        f'<div class="{classe}" style="background:{color_fons}; color:{color_text};">'
        f'<div class="nom">{index["nom"]}</div><div class="valor">{valor_txt}</div></div>'
    )


def _bloc_html(titol: str, variables: list, fila: dict, fila_comarca: Optional[dict], fila_provincia: Optional[dict], nom_comarca: str, nom_provincia: str) -> str:
    capcalera = (
        "<thead><tr><th></th>"
        f'<th class="valor">{fila["municipi"]}</th>'
        f'<th class="valor">{nom_comarca}</th>'
        f'<th class="valor">{nom_provincia}</th></tr></thead>'
    )
    files_html = []
    for col, etiqueta, format_tipus in variables:
        valor = fila.get(col)
        valor_comarca = (fila_comarca or {}).get(col)
        valor_provincia = (fila_provincia or {}).get(col)
        classe = lambda v: "valor sense-dada" if _es_nan(v) else "valor"
        cercle = _cercle_semafor_html(fila.get(f"semafor_{col}"))
        files_html.append(
            f"<tr><td>{etiqueta}</td>"
            f'<td class="{classe(valor)}">{cercle}{formata_valor(valor, format_tipus)}</td>'
            f'<td class="{classe(valor_comarca)}">{formata_valor(valor_comarca, format_tipus)}</td>'
            f'<td class="{classe(valor_provincia)}">{formata_valor(valor_provincia, format_tipus)}</td></tr>'
        )
    return f'<div class="custom-box">{titol}</div><table>{capcalera}<tbody>{"".join(files_html)}</tbody></table>'


def render_html(municipi: str) -> str:
    """Fragment HTML autocontingut (amb <style> inclòs, classes prefixades
    `fitxa-dp-*` per no col·lidir amb main.css), llest per
    `st.markdown(html, unsafe_allow_html=True)`. Sense targeta blanca: s'integra
    directament a la pàgina, reutilitzant `.custom-box` i l'estil de taula ja
    existents a main.css."""
    fila, comarca, provincia, fila_comarca, fila_provincia = _fila_i_comparatives(municipi)
    if fila is None:
        return ""
    nom_comarca = comarca or "Comarca"
    nom_provincia = provincia or "Província"
    resum = "".join(_caixa_index_html(idx) for idx in resum_indexs(municipi))
    blocs = "".join(
        _bloc_html(titol, variables, fila, fila_comarca, fila_provincia, nom_comarca, nom_provincia)
        for titol, variables in _TOTS_ELS_BLOCS
    )
    return f"""<style>{_CSS_FITXA}</style>
<div class="fitxa-dp-contenidor">
<h1>{fila['municipi']}</h1>
<div class="fitxa-dp-subtitol">{nom_comarca} · {nom_provincia}</div>
<div class="fitxa-dp-resum">{resum}</div>
{blocs}
<div class="fitxa-dp-metodologia">{text_metodologia()}</div>
</div>"""
