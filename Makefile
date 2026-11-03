VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(PYTHON) -m pip

.PHONY: venv install run clean

venv:
	python3 -m venv $(VENV)

install: venv requirements.txt
	$(PIP) install -r requirements.txt

run: 
	$(PYTHON) src/machine_learning/main.py

clean:
	rm -rf $(VENV)