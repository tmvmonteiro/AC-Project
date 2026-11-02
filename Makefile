VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all setup install clean

all: setup install run

setup:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: setup
	@$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV) __pycache__ *.pyc