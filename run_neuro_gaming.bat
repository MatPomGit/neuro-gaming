@echo off
setlocal

REM Uruchamia aplikację NeuroGaming z katalogu projektu.
cd /d "%~dp0"

REM Jeżeli istnieje lokalny venv, aktywuj go automatycznie.
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Start aplikacji Kivy.
python main.py

endlocal
