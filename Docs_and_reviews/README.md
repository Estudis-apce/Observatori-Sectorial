# Docs and reviews

Material de apoyo del Observatori Sectorial APCE. Reorganizado el 21/09/2026 mediante
movimientos; se conservan los archivos. La aplicación y sus datos activos permanecen
en la raíz del proyecto y en Resources/, respectivamente.

## Qué contiene

| Carpeta | Contenido |
|---|---|
| [docs/](docs/) | Documentación Word, comparativa PDF e instrucciones CLAUDE.md conservadas. |
| [_revisio_calculs/](_revisio_calculs/) | Decisiones, planes, diagnósticos y scripts de revisión. |
| [historic/](historic/) | Código antiguo, copias de datos y entregas de Atlas conservadas. |

La organización interna de historic/ se conserva:
codi_antic/, dades_duplicades/ y entrades_atlas/.
También se conserva allí `__pycache__/`, trasladada desde la raíz sin borrar su contenido.
El lanzador `Obrir_app.bat` usa `python -B` para no volver a escribir caché de módulos.
Abrir Python por otra vía sin esa opción puede regenerar la carpeta original.

## Por dónde empezar

1. [Decisiones de cálculo](_revisio_calculs/DECISIONS_PENDENTS.md): criterios ya acordados.
2. [Guía vigente de promociones](_revisio_calculs/PROPOSTA_NUM_PROMOCIONS_VIGENT.md):
   criterio actual, ubicaciones, cálculo y pruebas. El plan sin sufijo se conserva como histórico.
3. [Diagnóstico del mapa de viviendas](_revisio_calculs/DIAGNOSTICO_MAPA_VIVIENDAS.md).
4. Optimización de los mapas folium:
   [fase 1](_revisio_calculs/OPTIMIZACION_MAPAS_FASE1.md) (diagnóstico, mapas más ligeros) y
   [fase 2](_revisio_calculs/OPTIMIZACION_MAPAS_FASE2.md) (mapas cacheados, `st.fragment`, y zoom
   y agrupación del mapa de viviendas).
5. Otros planes y diagnósticos del mismo directorio, según el encargo.

La antigua propuesta de orden de carpetas se conserva como antecedente y lleva una
nota con la ubicación vigente. No ejecutar automáticamente sus fases antiguas.

## Scripts

Los scripts siguen juntos en _revisio_calculs/scripts/ para conservar sus imports.
base.py es el banco común y verificacio_calculs.py la batería general.
Antes de ejecutar cualquier script, revisar qué hace. Los p_* aplicaron cambios:
no volver a ejecutarlos como si fueran pruebas.

Desde la raíz del proyecto, la ruta de acceso actual es:

~~~text
cd Docs_and_reviews/_revisio_calculs/scripts
~~~

## Datos e históricos

Los archivos que alimentan la web siguen en Resources/JSON/. Los Excel de
historic/entrades_atlas/ son copias históricas y no los utiliza la app en su funcionamiento
habitual. La copia de promociones 260915 es anterior a la entrega corregida 260916.

El generador vigente consultado está fuera de esta carpeta, en:
Z:/ESTUDIS/Estudi d'oferta/2026/BBDD_v1/BBDD_v1.ipynb.
Sus salidas se guardan en BBDD_v1/outputs/. No ejecutarlo ni reemplazar salidas como
parte de una simple reorganización de archivos.

## Conservación y Git

No borrar archivos ni carpetas sin autorización expresa del usuario.
No sobrescribir un destino existente: conservar las dos versiones y resolver la colisión.

_revisio_calculs/ e historic/ se excluyen de Git mediante reglas específicas del proyecto.
docs/ y este índice no se excluyen por esta reorganización. No se ha creado ningún commit.

Si un documento o comentario antiguo menciona rutas anteriores, estas son las equivalencias:

| Ruta anterior desde la raíz | Ruta actual |
|---|---|
| docs/ | Docs_and_reviews/docs/ |
| _revisio_calculs/ | Docs_and_reviews/_revisio_calculs/ |
| arxiu/ | Docs_and_reviews/historic/ |

Se han actualizado las referencias operativas de los Markdown de decisiones, revisión y
promociones. Las referencias históricas en copias antiguas y comentarios del código se
conservan; no se ha modificado el código de la app ni de los scripts.
