VENV  = .venv
BIN   = $(VENV)/bin

.PHONY: test

test:
	$(BIN)/pytest -v
