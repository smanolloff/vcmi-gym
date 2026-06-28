MAKEFLAGS += --always-make
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -euxo pipefail -c

help:
	@echo "Tasks:"
	@grep -E '^[a-zA-Z][a-zA-Z0-9_.-]*:.*?' $(MAKEFILE_LIST) \
		| awk -F':' '{printf "  \033[36m%-20s\033[0m\n", $$1}' \
		| uniq

_require-pip-tools:
	which pip-compile || pip install pip-tools

_require-python-venv:
	test -n "$${VIRTUAL_ENV:-}" -o -n "$${CONDA_PREFIX:-}"

# XXX: VastAI containers should have those according to the image used.
# The CPU default is a fallback for local development.
PYTORCH_VERSION ?= 2.12.1
PYTORCH_BACKEND ?= cpu
TORCH = $(PYTORCH_VERSION)+$(PYTORCH_BACKEND)

pip-compile: _require-pip-tools
pip-compile: _require-python-venv
pip-compile:
	pip-compile \
		--strip-extras \
		--no-build-isolation \
		--pip-args="--only-binary=:all:" \
		--constraint=<(echo "torch==$(TORCH)") \
		--find-links="https://data.pyg.org/whl/torch-$(TORCH).html" \
		--output-file=requirements/requirements-torch-$(TORCH).txt \
		requirements/requirements.in

pip-install: _require-python-venv
pip-install:
	pip install -r requirements/requirements-torch-$(TORCH).txt

build-connector:
	cd vcmi_gym/connectors/ \
	&& cmake --preset vcmigym-rel \
	&& cmake --build rel -- -j8 \
	&& cd ../../

build-connector-debug:
	cd vcmi_gym/connectors/ \
	&& cmake --preset vcmigym-build \
	&& cmake --build build -- -j8 \
	&& cd ../../

vastai-build-connector:
	cd vcmi_gym/connectors/ \
	&& cmake -S . -B rel -Wno-dev \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_EXPORT_COMPILE_COMMANDS=0 \
	&& cmake --build rel/ -- -j$$(nproc)
