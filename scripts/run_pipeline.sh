#!/bin/sh
set -e
cd "$(dirname "$0")/../arm/python"
exec python3 pipeline.py "$@"
