# Convenience targets. The Python package builds the C library lazily on first
# use; `make lib` just forces that build (and prints the compiler commands).
PY ?= python3

.PHONY: lib install test test-fallback bench bench-quick report clean

lib:
	$(PY) -m lowbit.cli build --force

install:
	$(PY) -m pip install -e ".[test,bench]"

test:
	$(PY) -m pytest -q

test-fallback:
	LOWBIT_ISA=avx2 $(PY) -m pytest -q
	LOWBIT_ISA=scalar $(PY) -m pytest -q

bench:
	$(PY) -m lowbit.cli bench --out results --readme README.md --wait-idle 0.9

bench-quick:
	$(PY) -m lowbit.cli bench --quick --out /tmp/lowbit-quick

report:
	$(PY) -m lowbit.cli report --results results --readme README.md

clean:
	rm -rf src/lowbit/_build build .pytest_cache
