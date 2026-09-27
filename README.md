# Observatori sectorial APCE

Aplicación Streamlit de indicadores del sector residencial y del Estudi d'Oferta de obra nueva.

Para abrirla en local, utilizar `Obrir_app.bat` desde esta carpeta; requiere Python y las
dependencias de `requirements.txt`. La aplicación principal es `APP_Dades.py` y sus estilos
están en `main.css`.

## Carpetas de trabajo

- `Resources/`: JSON activos, portadas y logos que consume la aplicación. Se conservan sus rutas.
- [Docs_and_reviews/](Docs_and_reviews/README.md): documentación, planes de revisión,
  banco de pruebas y archivos históricos conservados.

Los datos que se actualizan para alimentar la web están en `Resources/JSON/`:
`DT_simple.json`, `DT_indicadors_demanda_potencial.json`, `BBDD_Atlas_trimmed.json`
y `BBDD_Promociones.json`. La carpeta contiene también la base geográfica municipal y
los tres JSON de la ficha de demanda potencial, actualmente desactivada.

Las carpetas técnicas de Git y Streamlit se mantienen en su ubicación.
La caché Python anterior se conserva en `Docs_and_reviews/historic/__pycache__/`.
`Obrir_app.bat` utiliza `python -B` para no escribir caché de módulos. Otros lanzamientos
de Python sin esa opción podrían volver a crear `__pycache__/` en la raíz.
No borrar archivos o carpetas ni sobrescribir destinos al reorganizar, salvo autorización expresa.
