#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Debugging
set -v
set -e
set -x

for name in experiment_name; do
  python -m src.main eval_only=true experiment.name=$name
done
