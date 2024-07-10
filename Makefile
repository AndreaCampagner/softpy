VENV := .venv/bin/activate
NMODULE := softpy

venv:
	. $(VENV) 
	pip install -r requirements.txt

# run:
# 	./actions.sh run

# test:
# 	./actions.sh test

verify: venv
	pylint $(NMODULE)