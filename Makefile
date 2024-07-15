NMODULE := softpy
PTESTS := tests

verify:
	pylint $(NMODULE)

tests:
	pytest -q $(PTESTS)