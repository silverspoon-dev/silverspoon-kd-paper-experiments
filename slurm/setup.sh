#!/bin/bash
# Install dependencies for the experiments.
# Run once on your cluster before launching SLURM jobs.

set -euo pipefail

pip install -r requirements.txt

echo "Setup complete. Run experiments with: bash scripts/<experiment>/<script>.sh"
