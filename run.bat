cd web
CALL ..\venv\Scripts\activate.bat
set secret_key=very_secret_key
set FLASK_APP=web-app.py
flask run
pause 