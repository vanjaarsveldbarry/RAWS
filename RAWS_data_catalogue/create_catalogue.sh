#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec pixi run python build.py
wait
exec pixi run python make_testcase.py


