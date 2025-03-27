cd gateway && sh bin/run.sh root/conf.yaml &
cd webapp && python3 -m venv venv && . venv/bin/activate && venv/bin/pip install -r requirements.txt
PYTHONPATH=. FLASK_APP=app:app python -m flask run --debug -p 5056 -h 0.0.0.0