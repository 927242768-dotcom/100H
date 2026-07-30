#!/bin/sh
set -e
python3 arm/python/install_rknnlite_from_wheel.py --clean
python3 scripts/check_bundle.py
