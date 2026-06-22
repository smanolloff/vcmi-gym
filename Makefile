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
	test -n $$VIRTUAL_ENV

# XXX: docker (i.e. vast) containers set these pytorch CUDA env vars accordingly

pip-compile: PYTORCH_VERSION ?= 2.11.0
pip-compile: PYTORCH_BACKEND ?= cpu
pip-compile: _require-pip-tools
pip-compile:
	pip-compile requirements.in -o requirements-$(PYTORCH_BACKEND).txt \
		--index-url https://pypi.org/simple \
		--extra-index-url https://download.pytorch.org/whl/$(PYTORCH_BACKEND) \
		--find-links https://data.pyg.org/whl/torch-$(PYTORCH_VERSION)+$(PYTORCH_BACKEND).html

pip-install: PYTORCH_BACKEND ?= cpu
pip-install: _require-python-venv
pip-install:
	pip install -r requirements-$(PYTORCH_BACKEND).txt

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
