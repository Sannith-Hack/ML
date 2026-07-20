@echo off
title HECAR Web Application Launcher
echo ========================================================
echo  Launching HECAR Web Application via Streamlit...
echo ========================================================
pip install -r requirements_web.txt
streamlit run web_app.py
pause
