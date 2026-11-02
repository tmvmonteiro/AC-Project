VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(PYTHON) -m pip

.PHONY: venv install notebook clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)