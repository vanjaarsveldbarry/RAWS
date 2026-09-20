#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pixi run python build.py
wait
pixi run python make_testcase.py


