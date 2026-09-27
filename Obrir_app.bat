@echo off
rem Obre l'Observatori sectorial en local (Streamlit) des d'aquesta carpeta.
rem Doble clic: arrenca el servidor i obre el navegador. Per aturar-lo, tanca aquesta finestra.
rem Les seccions en proves (Viabilitat Financera i Fitxa) nomes es veuen si existeix el fitxer
rem .seccions-en-proves en aquesta carpeta; sense ell, l'app es veu com a produccio.

title Observatori sectorial
cd /d "%~dp0"

python -B -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo No s'ha trobat Python amb Streamlit instal.lat.
    echo Instal.la'l amb:  python -m pip install streamlit
    pause
    exit /b 1
)

echo Arrencant l'Observatori sectorial... (el navegador s'obrira sol)
echo.
python -B -m streamlit run APP_Dades.py --browser.gatherUsageStats false

echo.
echo El servidor s'ha aturat.
pause
